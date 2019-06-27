# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import sys
import time

import torch
import torch.distributed as dist

from . import FairseqOptimizer


class FairseqBMUF(FairseqOptimizer):
    """
    Implements incremental block distributed data parallelism similar to
    https://ieeexplore.ieee.org/document/7472805

    Paper title: Scalable training of deep learning machines by incremental
    block training with intra-block parallel optimization and blockwise
    model-update filtering
    """

    def __init__(self, args, params, optimizer):

        super().__init__(args, params)
        self._optimizer = optimizer
        self.params = params
        self._num_updates = 0
        self.sync_iter = self.args.global_sync_iter
        self.block_momentum = 1 - 1.0 / self.args.distributed_world_size
        self.block_lr = self.args.block_lr
        self._reset_local_data()
        self.warmup_iteration = self.args.warmup_iterations
        self.use_nbm = self.args.use_nbm

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument(
            "--block-lr", default=1, type=float, help="block learning rate for bmuf"
        )
        parser.add_argument(
            "--global-sync-iter",
            default=10,
            type=int,
            help="Iteration for syncing global model",
        )
        parser.add_argument(
            "--warmup-iterations",
            default=500,
            type=int,
            help="warmup iterations for model to broadcast",
        )
        parser.add_argument(
            "--use-nbm",
            default=True,
            action="store_true",
            help="Specify whether you want to use classical BM / Nesterov BM",
        )

    @property
    def optimizer(self):
        return self._optimizer.optimizer

    @property
    def optimizer_config(self):
        return self._optimizer.optimizer_config

    def get_lr(self):
        return self._optimizer.get_lr()

    def set_lr(self, lr):
        self._optimizer.set_lr(lr)

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        self._optimizer.load_state_dict(state_dict, optimizer_overrides)

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        self._optimizer.multiply_grads(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        return self._optimizer.clip_grad_norm(max_norm)

    def _sync_block(self):
        if self.get_num_updates() % self.sync_iter == 0:
            if self.block_momentum != 0:
                self._BM_before_sync()

            self._allreduce_parameter()

            if self.block_momentum != 0:
                self._BM_after_sync()

    def _broadcast_model(self, rootRank=0):
        if (
            self.warmup_iteration != 0
            and self.get_num_updates() % self.warmup_iteration == 0
        ):
            self.warmup_iteration = 0

            # broadcast the local model
            for param in self.params:
                dist.broadcast(param.data, rootRank)

            # Also, broadcast the local parameters
            for param in (
                self.params_localprev
                + self.smoothed_grads_localprev
                + self.grads_localprev
            ):
                dist.broadcast(param, src=rootRank)

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._optimizer.step(closure)
        self.set_num_updates(self.get_num_updates() + 1)
        if self.warmup_iteration != 0:
            self._broadcast_model()
        else:
            self._sync_block()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self._optimizer.zero_grad()

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates

    @torch.no_grad()
    def _reset_local_data(self):
        self.params_localprev = [torch.zeros_like(p.data) for p in self.params]

        self.smoothed_grads_localprev = [
            p.data.new_zeros(p.data.size()) for p in self.params
        ]
        self.grads_localprev = [p.data.new_zeros(p.data.size()) for p in self.params]

        # initialize
        for param, copy_param in zip(self.params, self.params_localprev):
            copy_param.copy_(param.data)

    @torch.no_grad()
    def _BM_before_sync(self):
        # prev_param is basically the global copy from the previously finished
        # synchronisation. param.data is local parameter after block_sync_freq
        # for the local gpu. so grad is difference between previously synced
        # model and currrent local model.
        for index, (param, prev_param) in enumerate(
            zip(self.params, self.params_localprev)
        ):
            self.grads_localprev[index] = prev_param - param.data

    def _allreduce_parameter(self):
        for index, param in enumerate(self.params):
            sync_para = (
                param.data if self.block_momentum == 0 else self.grads_localprev[index]
            )
            sync_para /= float(dist.get_world_size())
            dist.all_reduce(sync_para, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def _BM_after_sync(self):
        for index, (param, prev_param, smoothed_grad, grad) in enumerate(
            zip(
                self.params,
                self.params_localprev,
                self.smoothed_grads_localprev,
                # all machines would share the same value of smoothed_grad, since it is
                # always computed on synchronized gradients.
                self.grads_localprev,
            )
        ):
            # prev_param is basically last syncrhornized parameter. though
            # smoothed_grad is local, all processes will have same value of
            # smoothed_grad and hence param is globally synchronized copy.
            # This is essentially a first-order infinite impulse response (IIR)
            # filter with the gain (1 - BM)*BM_lr:
            # smoothed_grad(t)=BM * smoothed_grad(t-1) + (1 - BM)*BM_lr*grad(t)
            smoothed_grad = (
                smoothed_grad * self.block_momentum
                + grad * (1 - self.block_momentum) * self.block_lr
            )
            param.data.copy_(prev_param - smoothed_grad)
            # A Nesterov momentum here is to do a partial weight update before
            # calculating the gradient
            if self.use_nbm:
                param.data.copy_(param.data - self.block_momentum * smoothed_grad)
            # backup for the next synchronization.
            self.smoothed_grads_localprev[index] = smoothed_grad
            prev_param.copy_(param.data)
