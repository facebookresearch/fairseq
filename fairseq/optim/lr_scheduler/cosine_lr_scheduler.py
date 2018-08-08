# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('cosine')
class CosineSchedule(FairseqLRScheduler):
    """Assign LR based on a cyclical schedule that follows the cosine function.

    See https://arxiv.org/pdf/1608.03983.pdf for details

      lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))

    where

      t_curr is current percentage of updates within the current period range
      t_i is the current period range, which is scaled by t_mul after every iteration

    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with cosine.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        self.min_lr = args.lr[0]
        self.max_lr = args.max_lr

        assert self.max_lr > self.min_lr, 'max_lr must be more than lr'

        self.t_mult = args.t_mult
        self.period = args.lr_period_updates
        self.lr_shrink = args.lr_shrink

        # initial learning rate
        self.lr = self.max_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--max-lr', required=True, type=float, metavar='LR',
                            help='max learning rate, must be more than args.lr')
        parser.add_argument('--t-mult', default=1, type=float, metavar='LR',
                            help='factor to grow the length of each period')
        parser.add_argument('--lr-period-updates', default=5000, type=float, metavar='LR',
                            help='initial number of updates per period')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.t_mult != 1:
            i = math.floor(math.log(1 - num_updates / self.period * (1 - self.t_mult), self.t_mult))
            t_i = self.t_mult ** i * self.period
            t_curr = num_updates - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
        else:
            i = math.floor(num_updates / self.period)
            t_i = self.period
            t_curr = num_updates - (self.period * i)

        lr_shrink = self.lr_shrink ** i
        min_lr = self.min_lr * lr_shrink
        max_lr = self.max_lr * lr_shrink

        self.lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))

        self.optimizer.set_lr(self.lr)
        return self.lr
