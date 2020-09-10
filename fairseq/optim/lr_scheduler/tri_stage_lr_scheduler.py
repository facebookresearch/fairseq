# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import register_lr_scheduler, LegacyFairseqLRScheduler
import math


@register_lr_scheduler('tri_stage')
class TriStageLRSchedule(LegacyFairseqLRScheduler):
    """Tristage learning rate schedulr

    Implement the learning rate scheduler in https://arxiv.org/pdf/1904.08779.pdf

    Similar to inverse_squre_root scheduler, but tri_stage learning rate employs
    three stages LR scheduling:

        - warmup stage, starting from `lr` * `init_lr_scale`, linearly
          increased to `lr` in `warmup_steps` iterations

        - hold stage, after `warmup_steps`, keep the LR as `lr` for `hold_steps`
          iterations

        - decay stage, after hold stage, decay LR exponetially to
          `lr` * `final_lr_scale` in `decay_steps`;
          after that LR is keep as `final_lr_scale` * `lr`

    During warmup::

      init_lr = args.init_lr_scale * args.lr
      lrs = torch.linspace(init_lr, args.lr, args.warmup_steps)
      lr = lrs[update_num]

    During hold::

      lr = args.lr

    During decay::

      decay_factor = - math.log(args.final_lr_scale) / args.decay_steps
      lr = args.lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)

    After that::

      lr = args.lr * args.final_lr_scale
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with tri-stage lr.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        # calculate LR at each point
        self.peak_lr = args.lr[0]
        self.init_lr = args.init_lr_scale * args.lr[0]
        self.final_lr = args.final_lr_scale * args.lr[0]

        # remember the steps at each stage
        self.warmup_steps = args.warmup_steps
        self.hold_steps = args.hold_steps
        self.decay_steps = args.decay_steps

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(args.final_lr_scale) / args.decay_steps

        # initial learning rate
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument(
            '--warmup-steps',
            default=4000,
            type=int,
            metavar='N',
            help='warmup the learning rate linearly for the first N updates'
        )
        parser.add_argument(
            '--hold-steps',
            default=20000,
            type=int,
            metavar='N',
            help='steps in hold stage.'
        )
        parser.add_argument(
            '--decay-steps',
            default=60000,
            type=int,
            metavar='N',
            help='steps in decay stages'
        )
        parser.add_argument(
            '--init-lr-scale',
            default=0.01,
            type=float,
            help="""
    initial learning rate scale during warmup phase; default is 0.01""")
        parser.add_argument(
            '--final-lr-scale',
            default=0.01,
            type=float,
            help="final learning rate scale; default to 0.01"
        )
        # fmt: on

    def _decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        stage, steps_in_stage = self._decide_stage(num_updates)
        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.optimizer.set_lr(self.lr)

        return self.lr
