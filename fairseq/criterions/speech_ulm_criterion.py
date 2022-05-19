# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from dataclasses import dataclass, field

import torch.nn.functional as F
from fairseq import metrics
from fairseq.tasks import FairseqTask
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SpeechUnitLmCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    loss_weights: str = field(
        default="1.;0.0;0.0",
        metadata={
            "help": "Weights of the losses that correspond to token, duration, and F0 streams"
        },
    )
    discrete_duration: bool = II("task.discrete_duration")
    discrete_f0: bool = II("task.discrete_f0")


def mae_loss(pred, targ, mask, reduce=True):
    if pred.ndim == 3:
        pred = pred.squeeze(2)
    else:
        assert pred.ndim == 2
    loss = (pred.float() - targ.float()).abs() * (~mask).float()
    loss = loss.sum() if reduce else loss.view(-1)
    return loss


def nll_loss(pred, targ, mask, reduce=True):
    lprob = F.log_softmax(pred, dim=-1)
    loss = F.nll_loss(lprob.view(-1, lprob.size(-1)), targ.view(-1), reduction="none")
    loss = loss * (~mask).float().view(-1)
    loss = loss.sum() if reduce else loss.view(-1)
    return loss


@register_criterion("speech_unit_lm_criterion", dataclass=SpeechUnitLmCriterionConfig)
class SpeechUnitLmCriterion(FairseqCriterion):
    def __init__(self, cfg: SpeechUnitLmCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.sentence_avg = cfg.sentence_avg
        self.weights = torch.tensor([float(w) for w in cfg.loss_weights.split(";")])
        assert self.weights.size(0) == 3
        assert (self.weights >= 0.0).all()

        self.dur_loss_fn = nll_loss if cfg.discrete_duration else mae_loss
        self.f0_loss_fn = nll_loss if cfg.discrete_f0 else mae_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        token_loss = nll_loss(
            net_output["token"], sample["target"], sample["mask"], reduce
        )
        dur_loss = self.dur_loss_fn(
            net_output["duration"],
            sample["dur_target"],
            sample["dur_mask"],
            reduce,
        )
        f0_loss = self.f0_loss_fn(
            net_output["f0"],
            sample["f0_target"],
            sample["f0_mask"],
            reduce,
        )
        loss = self.weights.to(token_loss.device) * torch.stack(
            [token_loss, dur_loss, f0_loss], dim=-1
        )
        loss = loss.sum() if reduce else loss.sum(-1)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.detach().sum().item(),
            "token_loss": token_loss.detach().sum().item(),
            "dur_loss": dur_loss.detach().sum().item(),
            "f0_loss": f0_loss.detach().sum().item(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        token_loss_sum = sum(log.get("token_loss", 0) for log in logging_outputs)
        dur_loss_sum = sum(log.get("dur_loss", 0) for log in logging_outputs)
        f0_loss_sum = sum(log.get("f0_loss", 0) for log in logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)

        metrics.log_scalar(
            "token_loss", token_loss_sum / sample_size, sample_size, round=3
        )

        metrics.log_scalar("dur_loss", dur_loss_sum / sample_size, sample_size, round=3)

        metrics.log_scalar("f0_loss", f0_loss_sum / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
