# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq.modules.quantization import pq, quantization_options, scalar


logger = logging.getLogger(__name__)


def quantize_model_scalar(model, args):
    quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    if quant_noise_scalar > 0:
        # quantize_model edits the model in place
        scalar.quantize_model_(model, p=quant_noise_scalar, bits=8, update_step=1000)
    return model


class Quantizer(object):
    def __init__(self, config_path, max_epoch, max_update):
        try:
            import yaml
        except ImportError:
            raise ImportError("Please install yaml with: pip install yaml")

        # parse config
        if config_path:
            with open(config_path) as config_file:
                config = quantization_options.parse_config_yaml(
                    yaml.safe_load(config_file)
                )
        else:
            config = quantization_options.parse_config_yaml({})

        self.n_centroids_config = config["n_centroids"]
        self.block_sizes_config = config["block_sizes"]
        self.layers_to_quantize = config["layers_to_quantize"]

        # We assume that training will run for a fixed number of epochs
        # (or updates) and that we should train for equal durations
        # between iterations of PQ.
        num_iterations = len(self.layers_to_quantize)
        if max_epoch > 0:
            assert max_epoch % num_iterations == 0, (
                "for iterative PQ, --max-epoch (={}) must be evenly divisible by "
                "len(layers_to_quantize) (={})".format(max_epoch, num_iterations)
            )
            self.epoch_schedule = max_epoch // num_iterations
        else:
            self.epoch_schedule = None
        if max_update > 0:
            assert max_update % num_iterations == 0, (
                "for iterative PQ, --max-update (={}) must be evenly divisible by "
                "len(layers_to_quantize) (={})".format(max_update, num_iterations)
            )
            self.update_schedule = max_update // num_iterations
        else:
            self.update_schedule = None
        assert (self.epoch_schedule is not None) ^ (
            self.update_schedule is not None
        ), "for iterative PQ, cannot specify both --max-update and --max-epoch"

        # 0 is a special value for quantization step, which will force
        # the first call to begin_epoch() to call step()
        self.quantization_step = 0

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.size_tracker = pq.SizeTracker(self.trainer.get_model())

    def step(self):
        """Move to the next stage of quantization."""
        if self.quantization_step >= len(self.layers_to_quantize):
            # Maybe we just finished the last training step or we loaded
            # a checkpoint for an iterative PQ model which previously
            # finished training. Either way, don't quantize again.
            return

        logger.info(
            "quantizing model (step={}; layers_to_quantize[step]={})".format(
                self.quantization_step, self.layers_to_quantize[self.quantization_step]
            )
        )
        quantized_layers = pq.quantize_model_(
            self.trainer.get_model(),
            self.size_tracker,
            self.layers_to_quantize,
            self.block_sizes_config,
            self.n_centroids_config,
            step=self.quantization_step,
        )
        logger.info("quantized layers: {}".format(quantized_layers))
        logger.info(self.size_tracker)

        self.quantization_step += 1

        # reintialize the Trainer since model parameters have changed
        self.trainer.reinitialize()

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch (epochs start at 1)."""
        if (
            (
                self.epoch_schedule is not None
                and epoch > 0
                and (epoch - 1) % self.epoch_schedule == 0
            )
            # we always step once in the beginning, even if using
            # update-based quantization
            or self.quantization_step == 0
        ):
            self.step()

    def step_update(self, num_updates):
        """Called at the end of each step."""
        if (
            self.update_schedule is not None
            and num_updates > 0
            and num_updates % self.update_schedule == 0
        ):
            self.step()

    def state_dict(self):
        return {
            "n_centroids_config": self.n_centroids_config,
            "block_sizes_config": self.block_sizes_config,
            "layers_to_quantize": self.layers_to_quantize,
            "epoch_schedule": self.epoch_schedule,
            "update_schedule": self.update_schedule,
            "quantization_step": self.quantization_step,
        }

    def load_state_dict(self, state_dict):
        self.n_centroids_config = state_dict["n_centroids_config"]
        self.block_sizes_config = state_dict["block_sizes_config"]
        self.layers_to_quantize = state_dict["layers_to_quantize"]
        self.epoch_schedule = state_dict["epoch_schedule"]
        self.update_schedule = state_dict["update_schedule"]
        self.quantization_step = state_dict["quantization_step"]
