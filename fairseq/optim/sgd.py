# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim

from typing import Iterable

from . import FairseqOptimizer, register_optimizer, optimizer_registry


@optimizer_registry.register('sgd')
class SGD(FairseqOptimizer):
    def __init__(self, params, lr: Iterable[float], momentum: float=0.0, weight_decay: float=0.0):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._optimizer = torch.optim.SGD(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
                            help='momentum factor')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.lr[0],
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return True
