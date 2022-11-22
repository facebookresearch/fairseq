# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss,
)


@dataclass
class SelfKDLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    K: int = field(
        default=4000,
        metadata={"help": "number of steps after which distillation begins"},
    )
    eta: float = field(
        default=1e-6,
        metadata={"help": "increment for alpha"}
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_self_kd", dataclass=SelfKDLabelSmoothedCrossEntropyCriterionConfig
)
class SelfKDLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        K,
        eta,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
        )
        self.K = K
        self.eta = eta
        self.sentence_avg = sentence_avg
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, update_num=None, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)
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
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, update_num, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=False
        )
        if update_num > self.K:
            w_t = model.decoder.embed_scale * model.decoder.embed_tokens(target)
            w_n = model.decoder.embed_scale * model.decoder.embed_tokens(lprobs.argmax(dim=-1))

            ## Yet to implement sigma ##

            q_n = torch.minimum(
                torch.full(loss.size(), 0.5, device="cuda"), 
                torch.exp(-2*torch.norm(w_t-w_n, dim=-1, keepdim=True))
            )
            p_n = lprobs.max(dim=-1, keepdim=True).values
            a = model.additional_params["alpha"]
            loss = (loss * (1.0 - a * q_n)) - (a * q_n * p_n)
            model.additional_params["alpha"] = min(1, a + self.eta)
        return loss.sum(), nll_loss.sum()