# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import logging
import torch


logger = logging.getLogger(__name__)


try:
    from megatron.model.fused_bias_gelu import bias_gelu_impl
    has_fused_bias_gelu = True
except ImportError:
    has_fused_bias_gelu = False


def load_megatron_fused_kernel():
    """Compile and load fused kernels from Megatron."""
    if getattr(load_megatron_fused_kernel, "has_run", False):
        return
    load_megatron_fused_kernel.has_run = True

    from megatron import fused_kernels
    from argparse import Namespace

    if not torch.distributed.is_initialized():
        args = Namespace(rank=0, masked_softmax_fusion=True)
        fused_kernels.load(args)
        return

    global_rank = torch.distributed.get_rank()
    args = Namespace(rank=global_rank, masked_softmax_fusion=True)

    # Always build on rank zero first.
    if global_rank == 0:
        logger.info("Compiling and loading fused kernels")
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)

    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()

    logger.info("Done with compiling and loading fused kernels.")


def fused_bias_gelu(x, bias):
    if not has_fused_bias_gelu:
        raise ImportError(
            "Cannot find fused Megatron kernels, please install Megatron from: "
            "github.com/NVIDIA/Megatron-LM"
        )
    load_megatron_fused_kernel()
    return bias_gelu_impl(x, bias)
