# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig
)
from fairseq.data.data_utils import lengths_to_mask


@dataclass
class LabelSmoothedCrossEntropyWithCtcCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    ctc_weight: float = field(default=1.0, metadata={"help": "weight for CTC loss"})


@register_criterion(
    "label_smoothed_cross_entropy_with_ctc",
    dataclass=LabelSmoothedCrossEntropyWithCtcCriterionConfig
)
class LabelSmoothedCrossEntropyWithCtcCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self, task, sentence_avg, label_smoothing, ignore_prefix_size,
            report_accuracy, ctc_weight
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size,
            report_accuracy
        )
        self.ctc_weight = ctc_weight

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ctc_loss = torch.tensor(0.).type_as(loss)
        if self.ctc_weight > 0.:
            ctc_lprobs, ctc_lens = model.get_ctc_output(net_output, sample)
            ctc_tgt, ctc_tgt_lens = model.get_ctc_target(sample)
            ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
            ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
            reduction = "sum" if reduce else "none"
            ctc_loss = F.ctc_loss(
                ctc_lprobs, ctc_tgt_flat, ctc_lens, ctc_tgt_lens,
                reduction=reduction, zero_infinity=True
            ) * self.ctc_weight
        loss += ctc_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "ctc_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
