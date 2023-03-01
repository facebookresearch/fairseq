# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss,
)


@dataclass
class RdropLabelSmoothedCrossEntropyCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    rdrop_alpha: float = field(
        default=0.0,
        metadata={"help": "alpha for r-drop, 0 means no r-drop"},
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_rdrop",
    dataclass=RdropLabelSmoothedCrossEntropyCriterionConfig,
)
class RdropLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        rdrop_alpha=0.0,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
        )
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.rdrop_alpha = rdrop_alpha

    def forward(self, model, sample, reduce=True, net_output=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if net_output is None:
            if self.rdrop_alpha > 0 and sample["net_input"]["src_tokens"].size(
                0
            ) == sample["target"].size(0):
                sample = duplicate_input(sample)
            net_output = model(**sample["net_input"])
        loss, nll_loss, rdrop_kl_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        if self.rdrop_alpha > 0:
            logging_output["rdrop_kl_loss"] = utils.item(rdrop_kl_loss.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.rdrop_alpha > 0 or target.size(0) != lprobs.size(0):
            target = torch.cat([target, target.clone()], dim=0)

        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        if self.rdrop_alpha > 0:
            pad_mask = target[: target.size(0) // 2].unsqueeze(-1).eq(self.padding_idx)
            rdrop_kl_loss = compute_kl_loss(model, net_output, pad_mask)
            loss += self.rdrop_alpha * rdrop_kl_loss
        else:
            rdrop_kl_loss = loss.new_zeros(1)
        return loss, nll_loss, rdrop_kl_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        rdrop_kl_loss = utils.item(
            sum(log.get("rdrop_kl_loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2)
        )
        if rdrop_kl_loss > 0:
            metrics.log_scalar("rdrop_kl_loss", rdrop_kl_loss)


def duplicate_input(sample):
    if "net_input" in sample.keys():
        sample_input = sample["net_input"]
    else:
        sample_input = sample

    for k, v in sample_input.items():
        if isinstance(v, torch.Tensor):
            sample_input[k] = torch.cat([v, v.clone()], dim=0)
    if "net_input" in sample.keys():
        sample["net_input"] = sample_input
    else:
        sample = sample_input
    return sample


def compute_kl_loss(model, net_output, pad_mask=None, reduce=True):
    net_prob = model.get_normalized_probs(net_output, log_probs=True)
    net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

    net_prob = net_prob.view(-1, net_prob.size(-1))
    net_prob_tec = net_prob_tec.view(-1, net_prob_tec.size(-1))

    p, q = torch.split(net_prob, net_prob.size(0) // 2, dim=0)
    p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0) // 2, dim=0)

    p_loss = torch.nn.functional.kl_div(p, q_tec, reduction="none")
    q_loss = torch.nn.functional.kl_div(q, p_tec, reduction="none")

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.0)
        q_loss.masked_fill_(pad_mask, 0.0)

    if reduce:
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
