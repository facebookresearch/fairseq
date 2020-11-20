# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

import torch.optim.lr_scheduler
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class ReduceLROnPlateauLRScheduleConfig(FairseqDataclass):
    lr_shrink: float = field(
        default=0.1, metadata={"help": "shrink factor for annealing"}
    )
    lr_threshold: float = field(
        default=1e-4,
        metadata={
            "help": (
                "threshold for measuring the new optimum, to only focus on "
                "significant changes"
            )
        },
    )
    lr_patience: int = field(
        default=0,
        metadata={
            "help": (
                "number of epochs with no improvement after which learning rate will "
                "be reduced"
            )
        },
    )
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
    lr: List[float] = II("optimization.lr")
    maximize_best_checkpoint_metric: bool = II(
        "checkpoint.maximize_best_checkpoint_metric"
    )


@register_lr_scheduler(
    "reduce_lr_on_plateau", dataclass=ReduceLROnPlateauLRScheduleConfig
)
class ReduceLROnPlateauLRSchedule(FairseqLRScheduler):
    """
    Decay the LR by a factor every time the validation loss plateaus.
    Also comes with optional warmup phase, where we linearly increase
    the learning rate from some initial learning rate
    (``--warmup-init-lr``) until the configured learning rate
    (``--lr``). Thereafter the lr is adjusted according to original
    reduce_on_plateau scheme.

    During warmup::

      lrs = torch.linspace(
          cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates
      )
      lr = lrs[update_num]
    """

    def __init__(self, cfg: ReduceLROnPlateauLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)
        if len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with reduce_lr_on_plateau."
                " Consider --lr-scheduler=fixed instead."
            )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer,
            patience=cfg.lr_patience,
            factor=cfg.lr_shrink,
            mode="max" if cfg.maximize_best_checkpoint_metric else "min",
            threshold=cfg.lr_threshold,
        )
        warmup_end_lr = cfg.lr[0]
        # if no warm up, sets initial lr to be cfg.lr[0]
        if cfg.warmup_init_lr < 0:
            cfg.warmup_init_lr = 0 if cfg.warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first cfg.warmup_updates
        if cfg.warmup_updates > 0:
            self.lr_step = (warmup_end_lr - cfg.warmup_init_lr) / cfg.warmup_updates

        # this flag is either set from arg when no warm up, or set by
        # step_update() when warmup finishes
        self.warmup_end = True if cfg.warmup_updates <= 0 else False

        # initial learning rate
        # this self.lr is used only during init and/or warm up period
        self.lr = cfg.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {
            "best": self.lr_scheduler.best,
            "last_epoch": self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.lr_scheduler.best = state_dict["best"]
        if "last_epoch" in state_dict:
            self.lr_scheduler.last_epoch = state_dict["last_epoch"]

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
        if self.cfg.warmup_updates > 0:
            if num_updates <= self.cfg.warmup_updates:
                self.lr = self.cfg.warmup_init_lr + num_updates * self.lr_step
                self.optimizer.set_lr(self.lr)
            else:
                if self.warmup_end is False:
                    self.warmup_end = True
        # else do nothing
        return self.optimizer.get_lr()
