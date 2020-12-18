# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import LegacyFairseqLRScheduler, register_lr_scheduler
import logging
import ast

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@register_lr_scheduler("manual")
class ManualSchedule(LegacyFairseqLRScheduler):
    """Decay the LR on a manual schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        self.epoch2lr = self.parse_manuallr_args(args.epoch2lr)
        self.update2lr = self.parse_manuallr_args(args.update2lr)
        logger.info("@@@ ManualSchedule epoch2lr={}".format(self.epoch2lr))
        logger.info("@@@ ManualSchedule update2lr={}".format(self.update2lr))

        if 1 in self.epoch2lr:
            self.lr = self.epoch2lr[1]
        elif 1 in self.update2lr:
            self.lr = self.update2lr[1]
        else:
            self.lr = args.lr[0]
        self.optimizer.set_lr(self.lr)  # Set the beginning of the epoch.

    def parse_manuallr_args(self, lr_args_str):
        lr_dict = ast.literal_eval(lr_args_str.replace(' ', ''))
        if not isinstance(lr_dict, dict):
            raise ValueError("epoch2lr/update2lr must be abel to evaluated to a dict")

        lr_args = {}
        logger.info("@@@ after parsing input dictionary lr_dict = {}".format(lr_dict))
        for key, val in lr_dict.items():
            if "," in key:
                for k in key.split(","):
                    lr_args[int(k)] = float(val)
            elif "-" in key:
                s = int(key.split("-")[0])
                e = int(key.split("-")[1])
                for k in range(s, e + 1, 1):
                    lr_args[k] = float(val)
            else:
                lr_args[int(key)] = float(val)

        return lr_args

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument(
            "--epoch2lr",
            type=str,
            metavar="DICT",
            default="{}",
            help="a dictionary used to set lr for each epoch manually",
        )
        parser.add_argument(
            "--update2lr",
            type=str,
            metavar="DICT",
            default="{}",
            help="a dictionary used to set lr for each update manually",
        )
        # fmt: on

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state_dict):
        if "lr" in state_dict:
            self.lr = state_dict["lr"]

    def get_next_lr(self, epoch):
        manual_keys = [k for k in self.epoch2lr if k <= epoch]
        if manual_keys:
            manual_lr = self.epoch2lr[max(manual_keys)]
        else:
            logger.warning("@@@ epoch={} does not exist in manual lr input. epoch2lr={}...".format(
                epoch, list(self.epoch2lr.items())[:min(10, len(self.epoch2lr.keys())-1)]
            ))
            manual_lr = self.optimizer.get_lr()
        return manual_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        manual_keys = [k for k in self.update2lr if k <= num_updates]
        if manual_keys:
            manual_lr = self.update2lr[max(manual_keys)]
        else:
            logger.warning("epoch={} does not exist in manual lr input update2lr={}...".format(
                num_updates, list(self.update2lr.items())[:min(10, len(self.update2lr.keys())-1)]))
            manual_lr = self.optimizer.get_lr()

        self.optimizer.set_lr(manual_lr)
        return self.optimizer.get_lr()
