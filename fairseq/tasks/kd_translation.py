# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file implements translation along with distillation
# Major portion of the code is similar to translation.py

from dataclasses import dataclass, field
import logging

from typing import Optional
from fairseq.tasks import register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks.translation import TranslationConfig, TranslationTask


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

@dataclass
class KDTranslationConfig(TranslationConfig):
    kd_strategy: str = field(
        default="word_and_seq_level", metadata={"help": "distillation strategy to be used"}
    )
    teacher_checkpoint_path: str = field(
        default="./", metadata={"help": "teacher checkpoint path when performing distillation"}
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
        self.kd_strategy = cfg.kd_strategy
        self.src_lang_ids = [i for i in range(len(src_dict)) if src_dict[i].startswith("__src__")]