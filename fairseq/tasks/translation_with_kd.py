# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional
import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    EVAL_BLEU_ORDER,
)
from fairseq.optim.amp_optimizer import AMPOptimizer


@dataclass
class KDTranslationConfig(TranslationConfig):
    teacher_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "teacher checkpoint path when performing distillation"},
    )
    language_tags: Optional[str] = field(
        default=None,
        metadata={"help": "language tags for Global-language-wise distillation"},
    )


@register_task("translation_with_kd", dataclass=KDTranslationConfig)
class KDTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: KDTranslationConfig

    def __init__(self, cfg: KDTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        if cfg.language_tags is not None:
            _rev_src_dict = {i: src_dict[i] for i in range(len(src_dict))}
            self.lang_ids = [_rev_src_dict[tag] for tag in cfg.language_tags.split(",")]
        else:
            self.lang_ids = None

    def train_step(
        self,
        sample,
        model,
        teacher_model,
        criterion,
        optimizer,
        update_num,
        ignore_grad=False,
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(
                    model, teacher_model, sample, update_num
                )
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, teacher_model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, teacher_model, sample)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output[f"_bleu_counts_{i}"] = bleu.counts[i]
                logging_output[f"_bleu_totals_{i}"] = bleu.totals[i]
        return loss, sample_size, logging_output
