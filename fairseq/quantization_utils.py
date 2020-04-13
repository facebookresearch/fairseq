# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules.quantization import pq, quantization_options, scalar


def quantize_model_scalar(model, args):
    quant_noise_scalar = getattr(args, 'quant_noise_scalar', 0)
    if quant_noise_scalar > 0:
        # quantize_model edits the model in place
        scalar.quantize_model_(model, p=quant_noise_scalar, bits=8, update_step=1000)
    return model


class Quantizer(object):

    def __init__(self, config_path, trainer, quantization_step=0):
        self.quantization_step = quantization_step
        self.trainer = trainer

        try:
            import yaml
        except ImportError:
            raise ImportError('Please install yaml with: pip install yaml')

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

        self.size_tracker = pq.SizeTracker(self.trainer.get_model())

    def step(self):
        """Move to the next stage of quantization."""
        quantized_layers = pq.quantize_model_(
            self.trainer.get_model(),
            self.size_tracker,
            self.layers_to_quantize,
            self.block_sizes_config,
            self.n_centroids_config,
            step=self.quantization_step,
        )
        self.quantization_step += 1

        # reintialize the Trainer since model parameters have changed
        self.trainer.reinitialize()

    def state_dict(self):
        # TODO
        return {
        }

    def load_state_dict(self, state_dict):
        # TODO
        pass

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        if (
            (
                self.epoch_schedule > 0
                and epoch > 0
                and epoch % self.epoch_schedule == 0
            )
            # we always step once in the beginning
            or self.quantization_step == 0
        ):
            self.step()

    def step_update(self, num_updates):
        """Called at the end of each step."""
        if (
            self.update_schedule > 0
            and num_updates > 0
            and num_updates % self.update_schedule == 0
        ):
            self.step()
