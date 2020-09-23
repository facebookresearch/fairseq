# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List

from fairseq.dataclass.utils import FairseqDataclass
from omegaconf import II

from . import FairseqLRScheduler, register_lr_scheduler


@dataclass
class CosineConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={
            "help": "initial learning rate during warmup phase; default is args.lr"
        },
    )
    max_lr: float = field(
        default=1.0, metadata={"help": "max learning rate, must be more than args.lr"}
    )
    t_mult: float = field(
        default=1.0, metadata={"help": "factor to grow the length of each period"}
    )
    lr_period_updates: float = field(
        default=-1, metadata={"help": "initial number of updates per period"}
    )
    lr_shrink: float = field(
        default=0.1, metadata={"help": "shrink factor for annealing"}
    )
    # TODO common var for parent class
    lr: List[float] = II("params.optimization.lr")
    max_update: int = II("params.optimization.max_update")


@register_lr_scheduler("cosine")
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
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with cosine."
                " Consider --lr-scheduler=fixed instead."
            )

        warmup_end_lr = args.max_lr
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = args.lr[0]

        self.min_lr = args.lr[0]
        self.max_lr = args.max_lr

        assert self.max_lr > self.min_lr, "max_lr must be more than lr"

        self.t_mult = args.t_mult
        self.period = args.lr_period_updates

        if self.period <= 0:
            assert (
                args.max_update >= 0
            ), "Either --max_update or --lr-period-updates must be set"
            self.period = args.max_update - args.warmup_updates

        if args.warmup_updates > 0:
            # linearly warmup for the first args.warmup_updates
            self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates
        else:
            self.lr_step = 1

        self.warmup_updates = args.warmup_updates
        self.lr_shrink = args.lr_shrink

        # initial learning rate
        self.lr = args.warmup_init_lr
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
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
        else:
            curr_updates = num_updates - self.args.warmup_updates
            if self.t_mult != 1:
                i = math.floor(
                    math.log(
                        1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult
                    )
                )
                t_i = self.t_mult ** i * self.period
                t_curr = (
                    curr_updates
                    - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
                )
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)

            lr_shrink = self.lr_shrink ** i
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            self.lr = min_lr + 0.5 * (max_lr - min_lr) * (
                1 + math.cos(math.pi * t_curr / t_i)
            )

        self.optimizer.set_lr(self.lr)
        return self.lr
