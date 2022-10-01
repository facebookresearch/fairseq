# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file implements translation along with distillation
# Major portion of the code is similar to translation.py

from dataclasses import dataclass, field
import logging

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

@dataclass
class KDTranslationConfig(TranslationConfig):
    # option to start knowledge distillation
    distillation_strategy: str = field(
        default="none", metadata={"help": "distillation strategy to be used"}
    )
    distillation_rate: float = field(
        default=0.5, metadata={"help": "the hyperparameter `tau` to control the number of words to get distillation knowledge"}
    )
    temperature_schedule: str = field(
        default="none", metadata={"help": "temperature schedule for distillation"}
    )
    temperature: float = field(
        default=1, metadata={"help": "initial temperature to be used during distillation"}
    )
    teacher_checkpoint_path: str = field(
        default="", metadata={"help": "teacher checkpoint path when performing distillation"}
    )
    difficult_queue_size: int = field(
        default=20000, metadata={"help": "queue size"}
    )


@register_task("kd_translation", dataclass=KDTranslationConfig)
class KDTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language along with knowledge distillation for seq2seq models
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The kd_translation task is compatible with :mod:`fairseq-train`
    """

    cfg: KDTranslationConfig

    def __init__(self, cfg: KDTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.distillation_strategy = cfg.distillation_strategy
        self.distillation_rate = cfg.distillation_rate
        self.temperature_schedule = cfg.temperature_schedule
        self.difficult_queue_size = cfg.difficult_queue_size
        self.temperature = cfg.temperature
