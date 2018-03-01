# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.optim.lr_scheduler

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('fixed')
class FixedSchedule(FairseqLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer.optimizer, self.anneal)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                            help='force annealing at specified epoch')

    def anneal(self, epoch):
        lrs = self.args.lr
        if self.args.force_anneal is None or epoch < self.args.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = lrs[-1] * self.args.lr_shrink ** (epoch + 1 - self.args.force_anneal)
        return next_lr / lrs[0]  # correct for scaling from LambdaLR

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        self.lr_scheduler.step(epoch)
        return self.optimizer.get_lr()
