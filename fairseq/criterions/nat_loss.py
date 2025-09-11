# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)



@dataclass
class LabelSmoothedDualImitationCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("nat_loss", dataclass=LabelSmoothedDualImitationCriterionConfig)
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.blank_index = getattr(task.target_dictionary, "blank_index", None)
        self.pad_idx = task.target_dictionary.pad()

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _compute_ctc_loss(self, logits, prev_output_tokens, targets, label_smoothing=0.0, name="ctc-loss", factor=1.0):
        """
        This method computes the ctc loss based on the logits using the pytorch ctc loss.
        It does the required transformations for the ctc loss and includes label smoothing.
        """
        target_mask = targets.ne(self.pad_idx)
        logit_mask = prev_output_tokens.ne(self.pad_idx)
        logit_lengths = (logit_mask.bool()).long().sum(1)

        if len(targets.size()) == 1:
            targets = targets.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0)
        target_lengths = (target_mask.bool()).long().sum(1)

        log_probs = logits.log_softmax(-1)  # B x T x N_classes
        log_probs_T = log_probs.transpose(0, 1)  # T x B x N_classes, this kind of shape is required for ctc_loss
        targets = targets[target_mask.bool()].long()
        loss = F.ctc_loss(
            log_probs_T.float(),
            targets,
            logit_lengths,
            target_lengths,
            blank=self.blank_index,
            reduction="mean",
            zero_infinity=True,
        )

        # The number of invalid samples are samples where the predictions are shorter than the targets.
        # If this is true for too many samples, one might think about increasing --ctc-src-upsample-scale.
        n_invalid_samples = (logit_lengths < target_lengths).long().sum()

        if n_invalid_samples > 0:
            logger.warning(
                f"The length of predicted alignment is shorter than target length, increase upsample factor: {n_invalid_samples} samples"
            )

        if label_smoothing > 0:
            smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
            loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss

        return {"name": name, "loss": loss, "factor": factor}
    
    def _compute_ds_ctc_loss(self, logits, prev_output_tokens, targets, label_smoothing=0.0, name="ds-ctc-loss", factor=1.0):
        logits = list(map(lambda layer_pred: layer_pred.squeeze(dim=0), torch.split(logits,1)))
        ctc_losses = list(map(lambda layer_pred: self._compute_ctc_loss(
            logits=layer_pred,
            prev_output_tokens=prev_output_tokens,
            targets=targets,
            label_smoothing=label_smoothing
        )['loss'], logits))

        return {"name": name, "loss": sum(ctc_losses) / len(ctc_losses), "factor": factor}

    def _compute_ds_loss(self, logits, targets, masks=None, label_smoothing=0.0, name="ds-loss", factor=1.0):
        logits = list(map(lambda layer_pred: layer_pred.squeeze(dim=0), torch.split(logits,1)))
        losses = list(map(lambda layer_pred: self._compute_loss(
            outputs=layer_pred,
            targets=targets,
            masks=masks,
            label_smoothing=label_smoothing
        ), logits))
        loss_mean = sum([loss['loss'] for loss in losses]) / len(losses)
        nll_loss_mean = sum([loss['nll_loss'] for loss in losses]) / len(losses)
        return {"name": name, "loss": loss_mean, "nll_loss": nll_loss_mean, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []

        for obj in outputs:
            if obj.startswith('glat'):
                continue
            if obj.startswith('ctc'):
                loss_function = self._compute_ctc_loss if not outputs[obj].get("deep_supervision") else self._compute_ds_ctc_loss
                _losses = loss_function(
                    outputs[obj].get("logits"),
                    outputs[obj].get("prev_output_tokens"),
                    outputs[obj].get("targets"),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            elif outputs[obj].get("loss", None) is None:
                loss_function = self._compute_loss if not outputs[obj].get("deep_supervision") else self._compute_ds_loss
                _losses = loss_function(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if "glat_accu" in outputs:
            logging_output["glat_accu"] = outputs['glat_accu']

        if "glat_context_p" in outputs:
            logging_output['glat_context_p'] = outputs['glat_context_p']
        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        glat_accu = utils.item(sum(log.get("glat_accu", 0) for log in logging_outputs))
        glat_context_p = utils.item(sum(log.get("glat_context_p", 0) for log in logging_outputs))

        metrics.log_scalar("loss", loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))
        metrics.log_scalar("glat_accu", glat_accu / sample_size, sample_size, round=3)
        metrics.log_scalar("glat_context_p", glat_context_p / sample_size, sample_size, round=3)

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
