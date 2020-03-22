# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('cosine')
class CosineSchedule(FairseqLRScheduler):
    """Assign LR based on a cyclical schedule that follows the cosine function.

    See https://arxiv.org/pdf/1608.03983.pdf for details.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    max learning rate (``--max-lr``).

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))

    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.

    If ``--max-lr`` is missing, we assume that ``--lr`` corresponds to
    the max learning rate.
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with cosine.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        self.warmup_updates = args.warmup_updates
        self.lr_shrink = args.lr_shrink

        self.min_lr = args.lr[0]
        if args.max_lr is not None:
            self.max_lr = args.max_lr
        else:
            self.min_lr = 0.0
            self.max_lr = args.lr[0]

        assert self.max_lr > self.min_lr, '--max-lr must be greater than --lr'

        warmup_end_lr = self.max_lr
        if args.warmup_init_lr >= 0.0:
            self.warmup_init_lr = args.warmup_init_lr
        else:
            self.warmup_init_lr = self.min_lr

        self.t_mult = args.t_mult
        if args.lr_period_updates > 0:
            self.period = args.lr_period_updates
        else:
            assert args.max_update > 0, 'Either --max-update or --lr-period-updates must be set'
            self.period = args.max_update - self.warmup_updates

        if self.warmup_updates > 0:
            # linearly warmup for the first *warmup_updates*
            self.lr_step = (warmup_end_lr - self.warmup_init_lr) / self.warmup_updates
        else:
            self.lr_step = 1

        # initial learning rate
        self.lr = self.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--max-lr', type=float, metavar='LR',
                            help='max learning rate, must be more than args.lr')
        parser.add_argument('--t-mult', default=1, type=float, metavar='LR',
                            help='factor to grow the length of each period')
        parser.add_argument('--lr-period-updates', default=-1, type=float, metavar='LR',
                            help='initial number of updates per period')
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            curr_updates = num_updates - self.warmup_updates
            if self.t_mult != 1:
                i = math.floor(math.log(1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult))
                t_i = self.t_mult ** i * self.period
                t_curr = curr_updates - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)

            lr_shrink = self.lr_shrink ** i
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            self.lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))

        self.optimizer.set_lr(self.lr)
        return self.lr
