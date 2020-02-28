# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim.lr_scheduler

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('reduce_lr_on_plateau')
class ReduceLROnPlateau(FairseqLRScheduler):
    """
    Decay the LR by a factor every time the validation loss plateaus.
    Also comes with optional warmup phase, where we linearly increase
    the learning rate from some initial learning rate
    (``--warmup-init-lr``) until the configured learning rate
    (``--lr``). Thereafter the lr is adjusted according to original
    reduce_on_plateau scheme.

    During warmup::

      lrs = torch.linspace(
          args.warmup_init_lr, args.lr, args.warmup_updates
      )
      lr = lrs[update_num]
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with reduce_lr_on_plateau.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, patience=0, factor=args.lr_shrink,
            threshold=args.lr_threshold)
        warmup_end_lr = args.lr[0]
        # if no warm up, sets initial lr to be args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        if args.warmup_updates > 0:
            self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates
        # this flag is either set from arg when no warm up, or set by
        # step_update() when warmup finishes
        self.warmup_end = True if args.warmup_updates <= 0 else False
        # initial learning rate
        # this self.lr is used only during init and/or warm up period
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing, lr_new = (lr * lr_shrink)')
        parser.add_argument('--lr-threshold', default=1e-4, type=float, metavar='LT',
                            help='Threshold for measuring the new optimum, \
                            to only focus on significant changes')
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        # fmt: on

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {
            'best': self.lr_scheduler.best,
            'last_epoch': self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.lr_scheduler.best = state_dict['best']
        if 'last_epoch' in state_dict:
            self.lr_scheduler.last_epoch = state_dict['last_epoch']

    def step(self, epoch, val_loss=None):
        """
        Update the learning rate at the end of the given epoch if warmup
        finishes otherwise no update of lr on epoch boundaries
        """
        if val_loss is not None and self.warmup_end is True:
            self.lr_scheduler.step(val_loss)
        else:
            self.lr_scheduler.last_epoch = epoch
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """
        Update the learning rate after each update."""
        # if there is warmup
        if self.args.warmup_updates > 0:
            if num_updates <= self.args.warmup_updates:
                self.lr = self.args.warmup_init_lr + num_updates*self.lr_step
                self.optimizer.set_lr(self.lr)
            else:
                if self.warmup_end is False:
                    self.warmup_end = True
        # else do nothing
        return self.optimizer.get_lr()
