# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from fairseq import registry
from fairseq.optim.lr_scheduler.fairseq_lr_scheduler import (  # noqa
    FairseqLRScheduler,
    LegacyFairseqLRScheduler,
)
from omegaconf import DictConfig


(
    build_lr_scheduler_,
    register_lr_scheduler,
    LR_SCHEDULER_REGISTRY,
    LR_SCHEDULER_DATACLASS_REGISTRY,
) = registry.setup_registry(
    "--lr-scheduler", base_class=FairseqLRScheduler, default="fixed"
)


def build_lr_scheduler(cfg: DictConfig, optimizer):
    return build_lr_scheduler_(cfg, optimizer)


# automatically import any Python files in the optim/lr_scheduler/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.optim.lr_scheduler." + file_name)
