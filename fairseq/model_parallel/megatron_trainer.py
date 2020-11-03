# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

from fairseq import distributed_utils
from fairseq.trainer import Trainer
from fairseq.dataclass.configs import FairseqConfig

try:
    from fairseq.model_parallel.megatron.mpu import (
        get_data_parallel_group,
        get_data_parallel_rank,
        get_data_parallel_world_size,
        get_model_parallel_group,
        get_model_parallel_src_rank,
        get_cuda_rng_tracker,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


class MegatronTrainer(Trainer):
    """Main class for model parallel with data parallel training."""

    def __init__(self, cfg: FairseqConfig, task, model, criterion, **kwargs):
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install the megatron submodule:"
                "\n\n  git submodule update --init "
                "fairseq/model_parallel/megatron"
            )
        super().__init__(cfg, task, model, criterion, **kwargs)

    @property
    def data_parallel_world_size(self):
        return get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        return get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        return get_model_parallel_src_rank() == 0

    def clip_grad_norm(self, clip_norm):
        def _aggregate_model_parallel_grad_norm(total_norm):
            total_norm = total_norm ** 2
            distributed_utils.all_reduce(total_norm, group=get_model_parallel_group())
            total_norm = total_norm ** 0.5
            return total_norm

        return self.optimizer.clip_grad_norm(
            clip_norm,
            aggregate_norm_fn=_aggregate_model_parallel_grad_norm,
        )

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        extra_state['rng_tracker_states'] \
            = get_cuda_rng_tracker().get_states()
        super().save_checkpoint(filename, extra_state)
    
    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        extra_state = super().load_checkpoint(filename, reset_optimizer=reset_optimizer, reset_lr_scheduler=reset_lr_scheduler, optimizer_overrides=optimizer_overrides, reset_meters=reset_meters)
        if extra_state is not None and 'rng_tracker_states' in extra_state:
            get_cuda_rng_tracker().set_states(
                extra_state['rng_tracker_states'])
        return extra_state