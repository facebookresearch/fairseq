# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Optional

import torch

from fairseq.dataclass.configs import DistributedTrainingConfig
from fairseq.distributed import utils as dist_utils


try:
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    has_FSDP = True
except ImportError:
    FSDP = torch.nn.Module
    has_FSDP = False


class FullyShardedDataParallel(FSDP):
    """
    A small wrapper around fairscale's FullyShardedDataParallel (FSDP) with some
    fairseq-specific checkpoint saving/loading logic.

    Args:
        use_sharded_state (bool): if True, then ``state_dict`` will return
            ``FSDP.local_state_dict`` and ``load_state_dict`` will call
            ``FSDP.load_local_state_dict``. Otherwise, ``state_dict`` will
            return the full model weights on data parallel rank 0 (empty on
            other ranks) and ``load_state_dict`` will broadcast model weights
            from rank 0 to other ranks.
    """

    def __init__(self, *args, use_sharded_state: bool = False, **kwargs):
        if not has_FSDP:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        super().__init__(*args, **kwargs)
        self.use_sharded_state = use_sharded_state

    @property
    def unwrapped_module(self) -> torch.nn.Module:
        if self.flatten_parameters:
            return self.module.module
        else:
            return self.module

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if self.use_sharded_state:
            return super().local_state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
        else:
            if self.rank == 0:
                return super().state_dict(
                    destination=destination, prefix=prefix, keep_vars=keep_vars
                )
            else:
                # We must call state_dict() due to use of communication
                # primitives. But we don't use the result.
                super().state_dict()
                return destination or {}

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        if self.use_sharded_state:
            return super().load_local_state_dict(state_dict, strict=strict)
        else:
            state_dict = dist_utils.broadcast_object(
                state_dict, src_rank=0, group=self.process_group
            )
            return super().load_state_dict(state_dict, strict=strict)


@contextlib.contextmanager
def fsdp_enable_wrap(cfg: DistributedTrainingConfig, use_sharded_state: bool = False):
    try:
        from fairscale.nn import enable_wrap
    except ImportError:
        raise ImportError(
            "Cannot find FullyShardedDataParallel. "
            "Please install fairscale with: pip install fairscale"
        )
    if cfg.memory_efficient_fp16:
        assert cfg.fp16  # memory_efficient_fp16 should imply fp16
    group = dist_utils.get_data_parallel_group()
    if group is None and cfg.distributed_world_size == 1:
        from fairscale.utils.testing import DummyProcessGroup
        group = DummyProcessGroup(rank=0, size=1)
    fsdp_config = {
        "process_group": group,
        "reshard_after_forward": not cfg.no_reshard_after_forward,
        "mixed_precision": cfg.fp16 and not cfg.memory_efficient_fp16,
        "fp32_reduce_scatter": cfg.fp32_reduce_scatter,
        "flatten_parameters": True,
        "cpu_offload": cfg.cpu_offload,
        "compute_dtype": torch.float16 if cfg.fp16 else torch.float32,
        "bucket_cap_mb": cfg.bucket_cap_mb,
    }
    with enable_wrap(
        wrapper_cls=FullyShardedDataParallel,
        use_sharded_state=use_sharded_state,
        **fsdp_config,
    ):
        yield


def fsdp_wrap(module, min_num_params: Optional[int] = None, **kwargs):
    """
    Helper to wrap layers/modules in FSDP. This falls back to a no-op if
    fairscale is not available.

    Args:
        module (nn.Module): module to (maybe) wrap
        min_num_params (int, Optional): minimum number of layer params to wrap
    """
    try:
        from fairscale.nn import wrap
        if min_num_params is not None:
            num_params = sum(p.numel() for p in module.parameters())
            if num_params >= min_num_params:
                return wrap(module, **kwargs)
            else:
                return module
        else:
            return wrap(module, **kwargs)
    except ImportError:
        return module
