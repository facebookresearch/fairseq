# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('polynomial_decay')
class PolynomialDecaySchedule(FairseqLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, optimizer, lr, warmup_updates, end_learning_rate, total_num_update, power, force_anneal):
        super().__init__(optimizer)

        self.lr = lr[0]
        if warmup_updates > 0:
            self.warmup_factor = 1. / warmup_updates
        else:
            self.warmup_factor = 1
        self.end_learning_rate = end_learning_rate
        self.total_num_update = total_num_update
        self.power = power
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        self.force_anneal = force_anneal

    @classmethod
    def from_args(cls, optimizer, args):
        return cls(optimizer, args.lr, getattr(args, 'warmup_updates', 0) or 0, args.end_learning_rate,
                   args.total_num_update, args.power, args.force_anneal)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                            help='force annealing at specified epoch')
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--end-learning-rate', default=0.0, type=float)
        parser.add_argument('--power', default=1.0, type=float)
        parser.add_argument('--total-num-update', default=1000000, type=int)

    def get_next_lr(self, epoch):
        lrs = self.lr
        if self.force_anneal is None or epoch < self.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.optimizer.get_lr()
        return next_lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            self.warmup_factor = num_updates / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif num_updates >= self.total_num_update:
            lr = self.end_learning_rate
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_learning_rate
            pct_remaining = 1 - (num_updates - warmup) / (self.total_num_update - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate
        self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()
