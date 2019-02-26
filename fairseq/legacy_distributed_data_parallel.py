# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
A modified version of the legacy DistributedDataParallel module that uses c10d
communication primitives. This is necessary for models that have conditional
computation (e.g., AdaptiveSoftmax) and which therefore do not work with the
c10d version of DDP.

This version also supports the *accumulate_grads* feature, which allows faster
training with `--update-freq`.
"""

import copy

import torch
from torch import nn
from torch.autograd import Variable

from . import distributed_utils


class LegacyDistributedDataParallel(nn.Module):
    """Implements distributed data parallelism at the module level.

    A simplified version of :class:`torch.nn.parallel.DistributedDataParallel`.
    This version uses a c10d process group for communication and does not
    broadcast buffers.

    Args:
        module (~torch.nn.Module): module to be parallelized
        world_size (int): number of parallel workers
        process_group (optional): the c10d process group to be used for
            distributed data all-reduction. If None, the default process group
            will be used.
        buffer_size (int, optional): number of elements to buffer before
            performing all-reduce (default: 256M).
    """

    def __init__(self, module, world_size, process_group=None, buffer_size=2**28):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.process_group = process_group

        # Never use a bigger buffer than the number of model params
        self.buffer_size = min(buffer_size, sum(p.numel() for p in module.parameters()))
        self.buffer = None

        # Flag used by the NCCL backend to make sure we only reduce gradients
        # one time in the execution engine
        self.need_reduction = False

        # We can also forcibly accumulate grads locally and only do the
        # all-reduce at some later time
        self.accumulate_grads = False

        # For NCCL backend, since every single NCCL call is asynchoronous, we
        # therefore directly enqueue all the NCCL reduction calls to the
        # default CUDA stream without spawning up other reduction threads.
        # This achieves the best performance.
        self._register_grad_hook()

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        return attrs

    def __setstate__(self, state):
        super().__setstate__(state)
        self._register_grad_hook()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _register_grad_hook(self):
        """
        This function registers the callback all-reduction function for the
        NCCL backend. All gradients will be all reduced in one single step.
        The NCCL reduction will directly be enqueued into the default CUDA
        stream. Therefore, no synchronization is needed.
        """

        def all_reduce(params):
            buffer = self.buffer
            nonzero_buffer = False
            if len(params) > 1:
                offset = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is not None:
                        buffer[offset:offset+sz].copy_(p.grad.data.view(-1))
                        nonzero_buffer = True
                    else:
                        buffer[offset:offset+sz].zero_()
                    offset += sz
            else:
                # we only have a single grad to all-reduce
                p = params[0]
                if p.grad is not None:
                    buffer = p.grad.data
                    nonzero_buffer = True
                elif p.numel() <= self.buffer.numel():
                    buffer = buffer[:p.numel()]
                    buffer.zero_()
                else:
                    buffer = torch.zeros_like(p)

            if nonzero_buffer:
                buffer.div_(self.world_size)

            distributed_utils.all_reduce(buffer, self.process_group)

            # copy all-reduced grads back into their original place
            offset = 0
            for p in params:
                sz = p.numel()
                if p.grad is not None:
                    p.grad.data.copy_(buffer[offset:offset+sz].view_as(p))
                else:
                    p.grad = buffer[offset:offset+sz].view_as(p).clone()
                offset += sz

        def reduction_fn():
            # This function only needs to be called once
            if not self.need_reduction or self.accumulate_grads:
                return
            self.need_reduction = False

            if self.buffer is None:
                self.buffer = next(self.module.parameters()).new(self.buffer_size)

            # All-reduce the gradients in buckets
            offset = 0
            buffered_params = []
            for param in self.module.parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                if param.grad.requires_grad:
                    raise RuntimeError("DistributedDataParallel only works "
                                       "with gradients that don't require "
                                       "grad")
                sz = param.numel()
                if sz > self.buffer.numel():
                    # all-reduce big params directly
                    all_reduce([param])
                else:
                    if offset + sz > self.buffer.numel():
                        all_reduce(buffered_params)
                        offset = 0
                        buffered_params.clear()
                    buffered_params.append(param)
                    offset += sz

            if len(buffered_params) > 0:
                all_reduce(buffered_params)

        # Now register the reduction hook on the parameters
        for p in self.module.parameters():

            def allreduce_hook(*unused):
                self.need_reduction = True
                Variable._execution_engine.queue_callback(reduction_fn)

            if p.requires_grad:
                p.register_hook(allreduce_hook)
