# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import ChoiceEnum, FairseqDataclass


_EPSILON = torch.finfo(torch.float32).eps
TARGET_DIST_NORM_CHOICES = ChoiceEnum(["none", "minmax"])


@dataclass
class KLDivergenceRerankingCriterionConfig(FairseqDataclass):
    target_dist_norm: TARGET_DIST_NORM_CHOICES = field(
        default="none",
        metadata={"help": "method to normalize the range of target scores"},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature in softmax for target distributions"},
    )
    forward_batch_size: int = field(
        default=32,
        metadata={
            "help": "number of hypotheses per batch for model forward (set a value smaller than --mt-beam to avoid OOM when training with a large beam size)"
        },
    )


@register_criterion(
    "kl_divergence_rereanking", dataclass=KLDivergenceRerankingCriterionConfig
)
class KLDivergenceRerankingCriterion(FairseqCriterion):
    def __init__(
        self, task, target_dist_norm, temperature, forward_batch_size,
    ):
        super().__init__(task)
        self.target_dist_norm = target_dist_norm
        self.temperature = temperature
        self.forward_batch_size = forward_batch_size

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        sample_size = sample["id"].numel()
        assert sample_size % self.task.cfg.mt_beam == 0, (
            f"sample_size ({sample_size}) cannot be divided by beam size ({self.task.cfg.mt_beam})."
            f"Please set --required-batch-size-multiple={self.task.cfg.mt_beam}."
        )

        # split into smaller batches for model forward
        batch_out = []
        for i in range(0, sample_size, self.forward_batch_size):
            j = min(i + self.forward_batch_size, sample_size)

            out = model(
                src_tokens=sample["net_input"]["src_tokens"][i:j, :],
                src_lengths=sample["net_input"]["src_lengths"][i:j],
            )

            batch_out.append(
                model.sentence_forward(out, sample["net_input"]["src_tokens"][i:j, :])
            )

        batch_out = torch.cat(batch_out, dim=0).view(
            self.task.cfg.mt_beam, sample_size // self.task.cfg.mt_beam, -1
        )  # T x B x C
        if model.joint_classification == "sent":
            batch_out = model.joint_forward(batch_out)
        scores = model.classification_forward(batch_out.view(sample_size, 1, -1)).view(
            -1, self.task.cfg.mt_beam
        )  # input: B x T x C

        loss = self.compute_kl_loss(
            scores, sample["target"][:, 0].view(-1, self.task.cfg.mt_beam)
        )

        sample_size = sample_size // self.task.cfg.mt_beam

        logging_output = {
            "loss": loss.detach(),
            "ntokens": sample["ntokens"],
            "nsentences": sample_size * self.task.cfg.mt_beam,
            "sample_size": sample_size,
            "scores": scores.detach(),
        }

        return loss, sample_size, logging_output

    def compute_kl_loss(self, logits, target):
        norm_target = target
        if self.target_dist_norm == "minmax":
            min_v = torch.min(target, 1, keepdim=True).values
            max_v = torch.max(target, 1, keepdim=True).values
            norm_target = (target - min_v) / (max_v - min_v + _EPSILON)

        target_dist = F.softmax(
            norm_target / self.temperature, dim=-1, dtype=torch.float32
        )
        model_dist = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = -(target_dist * model_dist - target_dist * target_dist.log()).sum()
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        loss = loss_sum / sample_size / math.log(2)
        metrics.log_scalar("loss", loss, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
