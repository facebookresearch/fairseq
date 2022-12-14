# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys
import torch

from typing import Optional
from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.logging import metrics

try:
    from ..data import MaeFinetuningImageDataset
except:
    sys.path.append("..")
    from data import MaeFinetuningImageDataset

logger = logging.getLogger(__name__)


@dataclass
class MaeImageClassificationConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    input_size: int = 224
    local_cache_path: Optional[str] = None

    rebuild_batches: bool = True


@register_task("mae_image_classification", dataclass=MaeImageClassificationConfig)
class MaeImageClassificationTask(FairseqTask):
    """ """

    cfg: MaeImageClassificationConfig

    @classmethod
    def setup_task(cls, cfg: MaeImageClassificationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        cfg = task_cfg or self.cfg

        self.datasets[split] = MaeFinetuningImageDataset(
            root=data_path,
            split=split,
            is_train=split == "train",
            input_size=cfg.input_size,
            local_cache_path=cfg.local_cache_path,
            shuffle=split == "train",
        )

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            if hasattr(actualized_cfg, "pretrained_model_args"):
                model_cfg.pretrained_model_args = actualized_cfg.pretrained_model_args

        return model

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if "correct" in logging_outputs[0]:
            zero = torch.scalar_tensor(0.0)
            correct = sum(log.get("correct", zero) for log in logging_outputs)
            metrics.log_scalar_sum("_correct", correct)

            metrics.log_derived(
                "accuracy",
                lambda meters: 100 * meters["_correct"].sum / meters["sample_size"].sum,
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize
