# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class CosineLRScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={
            "help": "initial learning rate during warmup phase; default is cfg.lr"
        },
    )
    lr: List[float] = field(
        default=II("optimization.lr"),
        metadata={"help": "max learning rate, must be more than cfg.min_lr"},
    )
    min_lr: float = field(default=0.0, metadata={"help": "min learning rate"})
    t_mult: float = field(
        default=1.0, metadata={"help": "factor to grow the length of each period"}
    )
    lr_period_updates: float = field(
        default=-1, metadata={"help": "initial number of updates per period"}
    )
    lr_shrink: float = field(
        default=0.1, metadata={"help": "shrink factor for annealing"}
    )
    # This is not required, but is for convenience in inferring lr_period_updates
    max_update: int = II("optimization.max_update")


@register_lr_scheduler("cosine", dataclass=CosineLRScheduleConfig)
class CosineLRSchedule(FairseqLRScheduler):
    """Assign LR based on a cyclical schedule that follows the cosine function.

    See https://arxiv.org/pdf/1608.03983.pdf for details.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    max learning rate (``--lr``).

    During warmup::

      lrs = torch.linspace(cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      lr = cfg.min_lr + 0.5*(cfg.lr - cfg.min_lr)*(1 + cos(t_curr / t_i))

    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """

    def __init__(self, cfg: CosineLRScheduleConfig, fairseq_optimizer):
        super().__init__(cfg, fairseq_optimizer)
        if isinstance(cfg.lr, Collection) and len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with cosine."
                f" Consider --lr-scheduler=fixed instead. ({cfg.lr})"
            )

        self.max_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr
        assert (
            self.max_lr > cfg.min_lr
        ), f"max_lr (={cfg.lr}) must be more than min_lr (={cfg.min_lr})"

        warmup_end_lr = self.max_lr
        if cfg.warmup_init_lr < 0:
            cfg.warmup_init_lr = cfg.min_lr

        self.t_mult = cfg.t_mult
        self.period = cfg.lr_period_updates

        if self.period <= 0:
            assert (
                cfg.max_update > 0
            ), "Either --max_update or --lr-period-updates must be set"
            self.period = cfg.max_update - cfg.warmup_updates

        if cfg.warmup_updates > 0:
            # linearly warmup for the first cfg.warmup_updates
            self.lr_step = (warmup_end_lr - cfg.warmup_init_lr) / cfg.warmup_updates
        else:
            self.lr_step = 1

        self.warmup_updates = cfg.warmup_updates
        self.lr_shrink = cfg.lr_shrink

        # initial learning rate
        self.lr = cfg.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.cfg.warmup_updates:
            self.lr = self.cfg.warmup_init_lr + num_updates * self.lr_step
        else:
            curr_updates = num_updates - self.cfg.warmup_updates
            if self.t_mult != 1:
                i = math.floor(
                    math.log(
                        1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult
                    )
                )
                t_i = self.t_mult**i * self.period
                t_curr = (
                    curr_updates
                    - (1 - self.t_mult**i) / (1 - self.t_mult) * self.period
                )
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)

            lr_shrink = self.lr_shrink**i
            min_lr = self.cfg.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            self.lr = min_lr + 0.5 * (max_lr - min_lr) * (
                1 + math.cos(math.pi * t_curr / t_i)
            )

        self.optimizer.set_lr(self.lr)
        return self.lr
