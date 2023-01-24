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
class FITranslationConfig(TranslationConfig):
    save_precision_matrices_to: Optional[str] = field(
        default="precision_matrices.pt", metadata={"help": "path to save fischer information parameters"}
    )


@register_task("translation_capture_fisher_information", dataclass=FITranslationConfig)
class FITranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """
 
    cfg: FITranslationConfig

    def __init__(self, cfg: FITranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.path = cfg.save_precision_matrices_to
        self._precision_matrices = {}

    def populate_precision_matrices(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self._precision_matrices[n] = torch.zeros_like(p)

    def save_precision_matrices(self):
        torch.save(self._precision_matrices, self.path)

    def normalize_precision_matrices(self):
        n_samples = len(self.datasets["train"])
        for _, p in self._precision_matrices.items():
            p /= n_samples

    def get_precision_matrices(self):
        return self._precision_matrices

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
            loss.detach()
        # update the precision matrices
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                self._precision_matrices[n] += (p.grad.data ** 2)
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        # do not run the optimizer
        pass