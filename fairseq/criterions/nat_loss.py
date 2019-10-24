# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
from fairseq import utils
import torch
from torch import Tensor

from . import FairseqCriterion, register_criterion


@register_criterion("nat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            '--label-smoothing',
            default=0.,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

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

        if not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (
                    1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss"):
        return {"name": name, "loss": loss, "factor": 1}

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
        losses = []
        if "mask_ins_out" in outputs:
            mask_ins_losses = self._compute_loss(
                outputs["mask_ins_out"],
                outputs["mask_ins_tgt"],
                outputs["mask_ins_mask"],
                name="m_ins-loss",
                factor=1 if "mask_ins_w" not in outputs else outputs["mask_ins_w"],
            )
            losses += [mask_ins_losses]

        if "word_ins_out" in outputs:
            word_ins_losses = self._compute_loss(
                outputs["word_ins_out"],
                outputs["word_ins_tgt"],
                outputs["word_ins_mask"],
                self.args.label_smoothing,
                name="w_ins-loss",
                factor=1 if "word_ins_w" not in outputs else outputs["word_ins_w"],
            )

            losses += [word_ins_losses]
            nll_loss = word_ins_losses["nll_loss"]

        if "word_del_out" in outputs:
            word_del_losses = self._compute_loss(
                outputs["word_del_out"],
                outputs["word_del_tgt"],
                outputs["word_del_mask"],
                0.01,
                name="w_del-loss",
                factor=1 if "word_del_w" not in outputs else outputs["word_del_w"],
            )

            losses += [word_del_losses]

        if "length_out" in outputs:
            length_losses = self._compute_loss(
                outputs["length_out"],
                outputs["length_tgt"],
                name="len-loss",
                factor=1 if "length_w" not in outputs else outputs["length_w"],
            )

            losses += [length_losses]

        for w in outputs:
            if "-loss" in w:
                losses += [self._custom_loss(outputs[w], w)]

        loss = sum(l["loss"] for l in losses)

        # NOTE: as we are summing up per token mlm loss and per sentence nsp loss
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss = sum(log.get("nll_loss", 0) for log in logging_outputs)

        results = {
            "loss": loss / sample_size / math.log(2) if sample_size > 0 else 0.0,
            "nll_loss": nll_loss / sample_size / math.log(2)
            if sample_size > 0
            else 0.0,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                results[key[:-5]] = (
                    sum(log.get(key, 0) for log in logging_outputs)
                    / sample_size
                    / math.log(2)
                    if sample_size > 0
                    else 0.0
                )

        return results
