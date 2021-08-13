# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import re
from glob import glob
from typing import Optional

import torch
from fairseq.dataclass.configs import DistributedTrainingConfig
from fairseq.distributed import utils as dist_utils
from fairseq.file_io import load_and_pop_last_optimizer_state

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
        is_moe (bool): if True, use MoE-specific checkpointing logic
        use_sharded_state (bool): if True, then ``state_dict`` will return
            ``FSDP.local_state_dict`` and ``load_state_dict`` will call
            ``FSDP.load_local_state_dict``. Otherwise, ``state_dict`` will
            return the full model weights on data parallel rank 0 (empty on
            other ranks) and ``load_state_dict`` will broadcast model weights
            from rank 0 to other ranks.
    """

    def __init__(self, *args, is_moe: bool = None, use_sharded_state: bool = False, **kwargs):
        if not has_FSDP:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        if is_moe is None:
            if torch.distributed.get_rank() == 0:
                from fairseq import pdb; pdb.set_trace()
            else:
                import time; time.sleep(1000)
        assert is_moe is not None
        super().__init__(*args, **kwargs)
        self.is_moe = is_moe
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
        elif self.is_moe:
            return super().state_dict(
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
        elif self.is_moe:
            return super().load_state_dict(state_dict, strict=strict)
        else:
            state_dict = dist_utils.broadcast_object(
                state_dict, src_rank=0, group=self.process_group
            )
            return super().load_state_dict(state_dict, strict=strict)


@contextlib.contextmanager
def fsdp_enable_wrap(cfg: DistributedTrainingConfig, use_sharded_state: bool = False, **kwargs):
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
        "state_dict_device": torch.device("cpu"),
        **kwargs
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


def consolidate_fsdp_shards(pth_prefix: str) -> str:
    if pth_prefix.endswith(".pt"):
        pth_prefix = pth_prefix[:-3]
    save_prefix = pth_prefix + "_consolidated"  # .pt'
    moe_paths = glob(f"{pth_prefix}*rank*shard*.pt")
    all_ckpt_files = sorted(glob(f"{pth_prefix}*shard*.pt"))
    assert all_ckpt_files, f"no paths matched {pth_prefix}*shard*.pt"
    weights = []
    metadata = []
    expert_paths = []
    expert_dest_paths = []
    expert_ranks = []
    dense = not bool(moe_paths)
    for p in all_ckpt_files:
        if re.search("rank-(\d+)", os.path.basename(p)):  # expert checkpoint
            expert_paths.append(p)
            r = re.search("rank-(\d+)", os.path.basename(p)).groups()[0]
            assert r not in expert_ranks
            expert_ranks.append(r)
            expert_dest_paths.append(f"{save_prefix}-rank-{r}.pt")
        else:
            ckpt = load_and_pop_last_optimizer_state(p)
            weights.append(ckpt["model"])
            metadata.append(ckpt["shard_metadata"])
    assert weights, f'all files were considered experts: {all_ckpt_files}'
    consolidated_weights = FSDP.consolidate_shard_weights(shard_weights=weights, shard_metadata=metadata, strict=False)
    del weights, metadata

    if dense:
        ckpt_consolidated = dict(
            model=consolidated_weights,
            cfg=ckpt["cfg"],
            extra_state=ckpt["extra_state"],
            optimizer_history=ckpt["optimizer_history"],
            args=ckpt["args"],
        )
        save_path = f"{save_prefix}.pt"
        torch.save(ckpt_consolidated, save_path)
        print(f"saved to {save_path}")
        return save_path

    ckpt_shared = dict(
        model=consolidated_weights,
        cfg=ckpt["cfg"],
        extra_state=ckpt["extra_state"],
        optimizer_history=ckpt["optimizer_history"],
        args=ckpt["args"],
    )
    torch.save(ckpt_shared, f"{save_prefix}-shared.pt")
    # Process experts
    for src, dst in zip(expert_paths, expert_dest_paths):
        ckpt = load_and_pop_last_optimizer_state(src)
        expert_wt = FSDP.consolidate_shard_weights(
            shard_weights=[ckpt["model"]], shard_metadata=[ckpt["shard_metadata"]], strict=False
        )
        full_ckpt = dict(
            model=expert_wt,
            cfg=ckpt["cfg"],
            extra_state=ckpt["extra_state"],
            optimizer_history=ckpt["optimizer_history"],
            args=ckpt["args"],
        )
        torch.save(full_ckpt, dst)
    print(f"saved consolidated MoE with prefix {save_prefix}.pt")
    return f"{save_prefix}.pt"
