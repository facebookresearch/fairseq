# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file implements translation along with distillation
# Major portion of the code is similar to translation.py

from dataclasses import dataclass, field
import logging
import torch 

from typing import Optional
from fairseq.tasks import register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks.translation import TranslationConfig, TranslationTask


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

@dataclass
class KDTranslationConfig(TranslationConfig):
    # option to start knowledge distillation
    distil_strategy: str = field(
        default="generic", metadata={"help": "distillation strategy to be used"}
    )
    distil_rate: float = field(
        default=0.5, metadata={"help": "the hyperparameter `tau` to control the number of words to get distillation knowledge"}
    )
    temp_schedule: Optional[str] = field(
        default=None, metadata={"help": "temperature schedule for distillation"}
    )
    student_temp: float = field(
        default=1, metadata={"help": "student model temperature for distillation"}
    )
    teacher_temp: float = field(
        default=1, metadata={"help": "teacher model emperature for distillation"}
    )
    teacher_checkpoint_path: str = field(
        default="./", metadata={"help": "teacher checkpoint path when performing distillation"}
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
        self.distil_strategy = cfg.distil_strategy
        self.distil_rate = cfg.distil_rate
        self.temp_schedule = cfg.temp_schedule
        self.teacher_temp = cfg.teacher_temp
        self.student_temp = cfg.student_temp
        self.difficult_queue_size = cfg.difficult_queue_size

    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, teacher_model=None
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
            teacher_model (~fairseq.models.BaseFairseqModel): th teacher model which is always set in evaluation mode

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, teacher_model=teacher_model)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    
    def valid_step(self, sample, model, criterion, teacher_model=None):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, teacher_model=teacher_model)
        return loss, sample_size, logging_output

