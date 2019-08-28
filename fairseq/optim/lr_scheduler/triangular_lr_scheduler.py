# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('triangular')
class TriangularSchedule(FairseqLRScheduler):
    """Assign LR based on a triangular cyclical schedule.

    See https://arxiv.org/pdf/1506.01186.pdf for details.
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with triangular.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        lr = args.lr[0]

        assert args.max_lr > lr, 'max_lr must be more than lr'
        self.min_lr = lr
        self.max_lr = args.max_lr
        self.stepsize = args.lr_period_updates // 2
        self.lr_shrink = args.lr_shrink
        self.shrink_min = args.shrink_min

        # initial learning rate
        self.lr = self.min_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--max-lr', required=True, type=float, metavar='LR',
                            help='max learning rate, must be more than args.lr')
        parser.add_argument('--lr-period-updates', default=5000, type=float, metavar='LR',
                            help='initial number of updates per period (cycle length)')
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing')
        parser.add_argument('--shrink-min', action='store_true',
                            help='if set, also shrinks min lr')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        cycle = math.floor(num_updates / (2 * self.stepsize))

        lr_shrink = self.lr_shrink ** cycle
        max_lr = self.max_lr * lr_shrink
        if self.shrink_min:
            min_lr = self.min_lr * lr_shrink
        else:
            min_lr = self.min_lr

        x = abs(num_updates / self.stepsize - 2 * (cycle + 1) + 1)
        self.lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))

        self.optimizer.set_lr(self.lr)
        return self.lr
