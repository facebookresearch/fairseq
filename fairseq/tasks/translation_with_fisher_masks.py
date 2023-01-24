# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional
import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.optim.amp_optimizer import AMPOptimizer


@dataclass
class FITranslationConfig2(TranslationConfig):
    fisher_masks_path: Optional[str] = field(
        default="masks", metadata={"help": "path to save fischer information parameters"}
    )


@register_task("translation_with_fisher_masks", dataclass=FITranslationConfig2)
class FITranslationTask2(TranslationTask):
    """
    Translate from one (source) language to another (target) language.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """
 
    cfg: FITranslationConfig2

    def __init__(self, cfg: FITranslationConfig2, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self._fisher_masks = torch.load(cfg.fisher_masks_path)

    def _apply_fisher_masks(self, model):
        for n, p in model.named_parameters():
            if n in self._fisher_masks:
                print(n, self._fisher_masks[n].sum(), p.grad.data.sum(), end=" ")
                p.grad.data.masked_fill_(self._fisher_masks[n], 0.0)
                print(p.grad.data.sum())

    def optimizer_step(self, optimizer, model, update_num):
        self._apply_fisher_masks(model)
        optimizer.step()