# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.optim

from . import FairseqOptimizer, register_optimizer, optimizer_registry

import argparse
from typing import Iterable


@optimizer_registry.register('adadelta')
class Adadelta(FairseqOptimizer):
    def __init__(self, params, lr: Iterable[float], adadelta_rho: float=0.9, adadelta_eps: float=1e-6,
                 weight_decay: float=0.0, anneal_eps: bool=False):
        super().__init__()
        self.lr = lr
        self.adadelta_rho = adadelta_rho
        self.adadelta_eps = adadelta_eps
        self.weight_decay = weight_decay
        self.anneal_eps = anneal_eps
        self._optimizer = torch.optim.Adadelta(params, **self.optimizer_config)

    @classmethod
    def from_args(cls, args: argparse.Namespace, params: Iterable[torch.nn.Parameter]):
        return cls(params, args.lr, args.adadelta_rho, args.adadelta_eps, args.weight_decay, args.anneal_eps)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adadelta-rho', type=float, default=0.9, metavar='RHO',
                            help='coefficient used for computing a running average of squared gradients')
        parser.add_argument('--adadelta-eps', type=float, default=1e-6, metavar='EPS',
                            help='term added to the denominator to improve numerical stability')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--anneal-eps', action='store_true', help='flag to anneal eps')
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
            'rho': self.adadelta_rho,
            'eps': self.adadelta_eps,
            'weight_decay': self.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return True
