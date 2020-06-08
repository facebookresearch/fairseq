# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

    def __init__(self, args, optimizer):

        super().__init__(args)
        self._optimizer = optimizer
        self._num_updates = 0
        self.sync_iter = self.args.global_sync_iter
        self.block_momentum = self.args.block_momentum
        self.block_lr = self.args.block_lr
        self._reset_local_data()
        self.warmup_iteration = self.args.warmup_iterations
        self.use_nbm = self.args.use_nbm
        self.initial_state = self._optimizer.state_dict()
        self.average_sync = self.args.average_sync
        self.world_size = self.args.distributed_world_size

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument(
            "--block-lr", default=1, type=float, help="block learning rate for bmuf"
        )
        parser.add_argument(
            "--block-momentum",
            default=0.875,
            type=float,
            help="block momentum for bmuf",
        )
        parser.add_argument(
            "--global-sync-iter",
            default=50,
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
            default=False,
            action="store_true",
            help="Specify whether you want to use classical BM / Nesterov BM",
        )
        parser.add_argument(
            "--average-sync",
            default=False,
            action="store_true",
            help="Specify whether you want to average the local momentum after each sync",
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
        self.initial_state = self._optimizer.state_dict()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        self._optimizer.multiply_grads(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        return self._optimizer.clip_grad_norm(max_norm, aggregate_norm_fn)

    def average_params(self):
        self._optimizer.average_params()

    def _block_sync(self):
        if self.world_size <= 1:
            return
        # Update the global model using local models from all GPUs
        # (Step-1) Calculate grad between previously synced model and
        # currrent local model
        if self.block_momentum != 0:
            self._calc_grad()

        # (Step-2) Average gradient from all GPUs
        self._avg_grad_from_all_gpus()

        # (Step-3) Calculate global momentum and update the global model
        if self.block_momentum != 0:
            self._update_global_model()

        # (Step-4) Average local optimizer params
        if self.average_sync:
            self.average_params()

    def _is_warmup_end(self):
        # Check whether train iterations is equal to warmup iter
        if self.get_num_updates() == self.warmup_iteration:
            return True
        return False

    def _is_bmuf_iter(self):
        # Check whether train iterations is equal to bmuf sync iter
        if (self.get_num_updates() > self.warmup_iteration) and (
            self.get_num_updates() % self.sync_iter == 0
        ):
            return True
        return False

    def _warmup_sync(self, root_rank=0):
        if self.world_size <= 1:
            return
        # Broadcast the local model to all gpus
        for param in self.params:
            dist.broadcast(param.data, src=root_rank)

        # Update local optimizer state
        if self.average_sync:
            self._optimizer.average_params()
        else:
            self._optimizer.load_state_dict(self.initial_state)

        self._reset_local_data()

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._optimizer.step(closure)
        self.set_num_updates(self.get_num_updates() + 1)
        if self._is_warmup_end():
            self._warmup_sync()
        elif self._is_bmuf_iter():
            self._block_sync()

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
        # (Step-0) Initialize global momentum parameters and store global copy on each gpu
        self.global_params = [torch.zeros_like(p.data) for p in self.params]
        self.smoothed_grads = [p.data.new_zeros(p.data.size()) for p in self.params]
        self.grads = [p.data.new_zeros(p.data.size()) for p in self.params]

        # saving the global model locally for calculating gradient during bmuf sync
        for param, global_param in zip(self.params, self.global_params):
            global_param.copy_(param.data)

    @torch.no_grad()
    def _calc_grad(self):
        # global_params is basically the global copy from the previously finished
        # synchronisation. param.data is local parameter after block_sync_freq
        # for the local gpu. so grad is difference between previously synced
        # model and currrent local model.
        for index, (param, global_param) in enumerate(
            zip(self.params, self.global_params)
        ):
            self.grads[index] = global_param - param.data

    def _avg_grad_from_all_gpus(self):
        for index, param in enumerate(self.params):
            sync_para = param.data if self.block_momentum == 0 else self.grads[index]
            sync_para /= float(dist.get_world_size())
            dist.all_reduce(sync_para, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def _update_global_model(self):
        for index, (param, global_param, smoothed_grad, grad) in enumerate(
            zip(
                self.params,
                self.global_params,
                self.smoothed_grads,
                # all gpus would share the same value of smoothed_grad, since it is
                # always computed on synchronized gradients.
                self.grads,
            )
        ):
            # global_param is basically last syncrhornized parameter. though
            # smoothed_grad is local, all processes will have same value of
            # smoothed_grad and hence param is globally synchronized copy.
            # smoothed_grad(t) = BM * smoothed_grad(t-1) + BM_lr * grad(t)
            smoothed_grad = self.block_momentum * smoothed_grad + self.block_lr * grad
            param.data.copy_(global_param - smoothed_grad)

            # A Nesterov momentum here is to do a partial weight update before
            # calculating the gradient
            if self.use_nbm:
                param.data.copy_(param.data - self.block_momentum * smoothed_grad)

            # backup for the next synchronization.
            self.smoothed_grads[index] = smoothed_grad
            global_param.copy_(param.data)
