"""
A modified version of the legacy DistributedDataParallel module that
uses c10d communication primitives. This is necessary for networks that
have conditional computation (e.g., AdaptiveSoftmax) and which therefore
do not work with the c10d version of DDP.
"""

import copy

import torch
from torch import nn
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
from torch.nn.parallel import DistributedDataParallel

from . import distributed_utils


class LegacyDistributedDataParallel(nn.Module):
    """Implements distributed data parallelism at the module level.

    A simplified version of torch.nn.parallel.DistributedDataParallel.
    This version uses a c10d process group for communication and does
    not broadcast buffers.

    Args:
        module: module to be parallelized
        world_size: number of parallel workers
        process_group (optional): the c10d process group to be used for
            distributed data all-reduction. If None, the default process
            group will be used.
        bucket_cap_mb: LegacyDistributedDataParallel will bucket
            parameters into multiple buckets so that gradient reduction
            of each bucket can potentially overlap with backward
            computation. bucket_cap_mb controls the bucket size in
            MegaBytes (MB) (default: 256)
    """

    def __init__(self, module, world_size, process_group=None, bucket_cap_mb=256):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.process_group = process_group

        # Flag used by the NCCL backend to make sure we only reduce gradients
        # one time in the execution engine
        self.need_reduction = False

        MB = 1024 * 1024
        # used for intra-node param sync and inter-node sync as well
        self.reduce_bucket_size = bucket_cap_mb * MB

        # For NCCL backend, since every single NCCL call is asynchoronous, we
        # therefore directly enqueue all the NCCL reduction calls to the
        # default CUDA stream without spawning up other reduction threads.
        # This achieves the best performance.
        self._register_grad_hook()

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        return attrs

    def __setstate__(self, state):
        super(DistributedDataParallel, self).__setstate__(state)
        self._register_grad_hook()

    def forward(self, *inputs, **kwargs):
        self.need_reduction = True
        return self.module(*inputs, **kwargs)

    def _register_grad_hook(self):
        """
        This function registers the callback all-reduction function for the
        NCCL backend. All gradients will be all reduced in one single step.
        The NCCL reduction will directly be enqueued into the
        default CUDA stream. Therefore, no synchronization is needed.
        """

        def reduction_fn():
            # This function only needs to be called once
            if not self.need_reduction:
                return

            self.need_reduction = False
            all_grads = []

            # Bucketing all the gradients
            for param in self.module.parameters():
                if not param.requires_grad:
                    continue
                if param.grad is not None and param.grad.requires_grad:
                    raise RuntimeError("DistributedDataParallel only works "
                                       "with gradients that don't require "
                                       "grad")
                if param.grad is not None:
                    # Adding the gradients for reduction
                    all_grads.append(param.grad.data)
                else:
                    all_grads.append(torch.zeros_like(param))

            # Now bucketing the parameters
            dev_grads_buckets = _take_tensors(all_grads,
                                              self.reduce_bucket_size)

            # Now reduce each bucket one after another
            for grads_batch in dev_grads_buckets:
                grads_batch_coalesced = _flatten_dense_tensors(grads_batch)

                grads_batch_coalesced /= self.world_size

                distributed_utils.all_reduce(grads_batch_coalesced, self.process_group)

                grads_batch_reduced = _unflatten_dense_tensors(grads_batch_coalesced, grads_batch)
                for grad, reduced in zip(grads_batch, grads_batch_reduced):
                    grad.copy_(reduced)

        # Now register the reduction hook on the parameters
        for p in self.module.parameters():
            if not p.requires_grad:
                continue

            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(reduction_fn)

            p.register_hook(allreduce_hook)