# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss

from fairseq.criterions import register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class GateRegLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    alpha: float = field(
        default=0.0,
        metadata={"help": "scaling factor for context gate regularization"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion(
    "label_smoothed_ce_ctxgate_reg", dataclass=GateRegLabelSmoothedCrossEntropyCriterionConfig
)
class GateRegLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        alpha,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size=ignore_prefix_size, report_accuracy=report_accuracy)
        self.alpha = alpha

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        ctx_pen = sum(net_output[1]["ctx_gates"]).sum(dim=0)
        if reduce:
            ctx_pen = ctx_pen.sum()
        return loss + self.alpha * ctx_pen, nll_loss
