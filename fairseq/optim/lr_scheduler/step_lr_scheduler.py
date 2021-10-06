# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class StepLRScheduleConfig(FairseqDataclass):
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
    lr_deacy_period: int = field(default=25000, metadata={"help": "decay period"})
    lr_decay: float = field(default=0.5, metadata={"help": "decay factor"})


@register_lr_scheduler("step", dataclass=StepLRScheduleConfig)
class StepLRSchedule(FairseqLRScheduler):
    """Decay learning rate every k updates by a fixed factor
    """

    def __init__(self, cfg: StepLRScheduleConfig, fairseq_optimizer):
        super().__init__(cfg, fairseq_optimizer)
        self.max_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr
        self.min_lr = cfg.min_lr
        self.lr_deacy_period = cfg.lr_deacy_period
        self.lr_decay = cfg.lr_decay
        self.warmup_updates = cfg.warmup_updates
        self.warmup_init_lr = (
            cfg.warmup_init_lr if cfg.warmup_init_lr >= 0 else self.min_lr
        )

        assert(self.lr_deacy_period > 0)
        assert(self.lr_decay <= 1)
        assert(self.min_lr >= 0)
        assert(self.max_lr > self.min_lr)

        if cfg.warmup_updates > 0:
            # linearly warmup for the first cfg.warmup_updates
            self.warmup_lr_step = (
                (self.max_lr - self.warmup_init_lr) / self.warmup_updates
            )
        else:
            self.warmup_lr_step = 1

        # initial learning rate
        self.lr = self.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.cfg.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.warmup_lr_step
        else:
            curr_updates = num_updates - self.cfg.warmup_updates
            lr_mult = self.lr_decay ** (curr_updates // self.lr_deacy_period)
            self.lr = max(self.max_lr * lr_mult, self.min_lr)

        self.optimizer.set_lr(self.lr)
        return self.lr
