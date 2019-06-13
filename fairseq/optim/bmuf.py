# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import torch.distributed as dist
from fairseq import optim

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

    def _model_average_step(self):
        if self.get_num_updates() % self.sync_iter == 0:
            size = float(dist.get_world_size())
            for p in self.params:
                dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
                p.data /= size

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._optimizer.step(closure)
        self.set_num_updates(self.get_num_updates() + 1)
        self._model_average_step()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self._optimizer.zero_grad()

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
