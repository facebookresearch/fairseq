# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round


@dataclass
class W2vBertCriterionConfig(FairseqDataclass):
    infonce: bool = field(
        default=False,
        metadata={
            "help": "if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss) for the w2v2 loss"
        },
    )
    w2v2_loss_weight: float = field(
        default=1.0,
        metadata={"help": "weight for the w2v2 loss"},
    )
    mlm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "weight for the mlm loss"},
    )
    mlm_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "label smoothing for mlm"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not w2v2 and mlm ones)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


@register_criterion("w2vbert", dataclass=W2vBertCriterionConfig)
class W2vBertCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        infonce=False,
        w2v2_loss_weight=1.0,
        mlm_loss_weight=1.0,
        mlm_label_smoothing=0.0,
        loss_weights=None,
        log_keys=None,
    ):
        super().__init__(task)
        self.infonce = infonce
        self.w2v2_loss_weight = w2v2_loss_weight
        self.mlm_loss_weight = mlm_loss_weight
        self.mlm_label_smoothing = mlm_label_smoothing
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        mlm_logits = model.get_mlm_logits(net_output).float()
        w2v2_logits = model.get_w2v2_logits(net_output).float()
        mlm_targets = model.get_mlm_targets(sample, net_output)
        w2v2_targets = model.get_w2v2_targets(sample, net_output)

        losses = []

        reduction = "none" if (not reduce) else "sum"
        if self.infonce:
            w2v2_loss = F.cross_entropy(w2v2_logits, w2v2_targets, reduction=reduction)
        else:
            w2v2_loss = F.binary_cross_entropy_with_logits(
                w2v2_logits, w2v2_targets.float(), weight=None, reduction=reduction
            )

        mlm_loss = F.cross_entropy(
            mlm_logits,
            mlm_targets,
            reduction=reduction,
            label_smoothing=self.mlm_label_smoothing,
        )

        loss = self.w2v2_loss_weight * w2v2_loss + self.mlm_loss_weight * mlm_loss
        losses.append(loss.detach().clone())

        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        elif "mask_indices" in sample["net_input"]:
            sample_size = sample["net_input"]["mask_indices"].sum()
        else:
            sample_size = (
                w2v2_targets.numel()
                if self.infonce
                else w2v2_targets.long().sum().item()
            )

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["mlm_logits"] = mlm_logits.cpu().numpy()
                    logging_output["w2v2_logits"] = w2v2_logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(model, "get_original_targets"):
                        original_target = model.get_original_targets(sample, net_output)
                    else:
                        original_target = mlm_targets
                    logging_output["mlm_targets"] = original_target.cpu().numpy()
                    logging_output["w2v2_targets"] = w2v2_targets.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                value = float(value)
                logging_output[lk] = value

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()
        logging_output["loss_w2v2"] = w2v2_loss.item()
        logging_output["loss_mlm"] = mlm_loss.item()

        # TODO: add mlm accuracy, percentage of codebook usage
        if self.infonce:
            with torch.no_grad():
                if w2v2_logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert w2v2_logits.dim() > 1, w2v2_logits.shape
                    max = w2v2_logits.argmax(-1) == 0
                    min = w2v2_logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = float(max.numel())

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "correct",
            "count",
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                if "lang_idx" in k:
                    continue
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss"):
                    metrics.log_scalar(
                        k, val / (sample_size or 1) / math.log(2), sample_size, round=3
                    )
                else:
                    metrics.log_scalar(k, val / len(logging_outputs), round=3)

    # FIXME: revert when gather based xla reduction is implemented
    # @staticmethod
    # def logging_outputs_can_be_summed() -> bool:
    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        # XXX: Gather based reduction not implemented for xla yet.
        # So we fall to sum based reduction for xla.
        return False
