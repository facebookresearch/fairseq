# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from fairseq import registry
from fairseq.criterions.fairseq_criterion import (  # noqa
    FairseqCriterion,
    LegacyFairseqCriterion,
)
from omegaconf import DictConfig


(
    build_criterion_,
    register_criterion,
    CRITERION_REGISTRY,
    CRITERION_DATACLASS_REGISTRY,
) = registry.setup_registry(
    "--criterion", base_class=FairseqCriterion, default="cross_entropy"
)


def build_criterion(cfg: DictConfig, task, from_checkpoint=False):
    return build_criterion_(cfg, task, from_checkpoint=from_checkpoint)


# automatically import any Python files in the criterions/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.criterions." + file_name)
