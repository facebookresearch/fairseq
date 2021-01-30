# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class TriangularLRScheduleConfig(FairseqDataclass):
    max_lr: float = field(
        default="???", metadata={"help": "max learning rate, must be more than cfg.lr"}
    )
    lr_period_updates: float = field(
        default=5000,
        metadata={"help": "initial number of updates per period (cycle length)"},
    )
    lr_shrink: float = field(
        default=0.1, metadata={"help": "shrink factor for annealing"}
    )
    shrink_min: bool = field(
        default=False, metadata={"help": "if set, also shrinks min lr"}
    )
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("triangular", dataclass=TriangularLRScheduleConfig)
class TriangularLRSchedule(FairseqLRScheduler):
    """Assign LR based on a triangular cyclical schedule.

    See https://arxiv.org/pdf/1506.01186.pdf for details.
    """

    def __init__(self, cfg: TriangularLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)
        if len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with triangular."
                " Consider --lr-scheduler=fixed instead."
            )

        lr = cfg.lr[0]

        assert cfg.max_lr > lr, "max_lr must be more than lr"
        self.min_lr = lr
        self.max_lr = cfg.max_lr
        self.stepsize = cfg.lr_period_updates // 2
        self.lr_shrink = cfg.lr_shrink
        self.shrink_min = cfg.shrink_min

        # initial learning rate
        self.lr = self.min_lr
        self.optimizer.set_lr(self.lr)

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
