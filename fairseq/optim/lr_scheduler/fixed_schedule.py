# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('fixed')
class FixedSchedule(FairseqLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, optimizer, warmup_updates, lr, force_anneal, lr_shrink):
        super().__init__(optimizer)

        self.warmup_updates = warmup_updates
        self.lrs = lr
        self.lr = lr[0]

        if warmup_updates > 0:
            self.warmup_factor = 1. / warmup_updates
        else:
            self.warmup_factor = 1

        self.force_anneal = force_anneal
        self.lr_shrink = lr_shrink

    @classmethod
    def from_args(cls, optimizer, args):
        return cls(optimizer, getattr(args, 'warmup_updates', 0) or 0, args.lr, args.force_anneal, args.lr_shrink)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                            help='force annealing at specified epoch')
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing, lr_new = (lr * lr_shrink)')
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        # fmt: on

    def get_next_lr(self, epoch):
        if self.force_anneal is None or epoch < self.force_anneal:
            # use fixed LR schedule
            next_lr = self.lrs[min(epoch, len(self.lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.lrs[-1] * self.lr_shrink ** (epoch + 1 - self.force_anneal)
        return next_lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.warmup_updates > 0 and num_updates < self.warmup_updates:
            self.warmup_factor = (num_updates + 1) / float(self.warmup_updates)
            self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()
