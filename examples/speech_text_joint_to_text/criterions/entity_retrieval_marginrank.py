# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class EntityRetrievalMarginRankCriterionConfig(FairseqDataclass):
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    margin: float = field(
        default=0.0,
        metadata={"help": "the margin to use"},
    )


@register_criterion(
    "entity_retrieval_marginrank", dataclass=EntityRetrievalMarginRankCriterionConfig
)
class EntityRetrievalMarginRankCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        report_accuracy=False,
        margin=0.0,
    ):
        super().__init__(task)
        self.report_accuracy = report_accuracy
        self.margin = margin

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss = None
        nretrievals = 0
        n_correct = 0
        for pr_out in net_output['positive_retrieval_out']:
            for nr_out in net_output['negative_retrieval_out']:
                _loss = F.margin_ranking_loss(
                    pr_out,
                    nr_out,
                    torch.ones_like(pr_out),
                    margin=self.margin,
                    reduction="sum" if reduce else "none")
                if loss is None:
                    loss = _loss
                else:
                    loss += _loss
                nretrievals += pr_out.shape[0]
                if self.report_accuracy:
                    n_correct += (pr_out > 0.0).sum().item()
                    n_correct += (nr_out < 0.0).sum().item()

        logging_output = {
            "loss": loss.data,
            "nretrievals": nretrievals,
            "nsentences": sample["target"].size(0),
            "sample_size": nretrievals,
        }
        if self.report_accuracy:
            logging_output["n_correct"] = n_correct
            logging_output["total"] = nretrievals * 2
        return loss, nretrievals, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
