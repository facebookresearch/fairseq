# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
#from examples.speech_recognition.criterions import CrossEntropyAcc
from examples.simultaneous_translation.utils.latency import LatencyTraining


@register_criterion("latency_augmented_cross_entropy_acc")
class LatencyAugmentedCrossEntropyWithAccCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.latency_weight = args.latency_weight
        self.latency_train = LatencyTraining(
            self.latency_weight,
            0,
            "differentiable_average_lagging",
            "average",
            args.mass_preservation
        )
    
    @staticmethod
    def add_args(parser):
        super(
            LatencyAugmentedCrossEntropyWithAccCriterion,
            LatencyAugmentedCrossEntropyWithAccCriterion
        ).add_args(parser)
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--latency-weight", default=0., type=float, metavar='D',
                            help="")

    def compute_loss(self, model, net_output, target, reduction, log_probs):
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs, target, ignore_index=self.padding_idx, reduction=reduction
        )
        return lprobs, loss

    def get_logging_output(self, sample, target, lprobs, loss):
        target = target.view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = (
            sample["target"].size(0) if self.args.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
            "nframes": torch.sum(sample["net_input"]["src_lengths"]).item(),
        }

        return sample_size, logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        """Computes the cross entropy with accuracy metric for the given sample.

        This is similar to CrossEntropyCriterion in fairseq, but also
        computes accuracy metrics as part of logging

        Args:
            logprobs (Torch.tensor) of shape N, T, D i.e.
                batchsize, timesteps, dimensions
            targets (Torch.tensor) of shape N, T  i.e batchsize, timesteps

        Returns:
        tuple: With three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training

        TODO:
            * Currently this Criterion will only work with LSTMEncoderModels or
            FairseqModels which have decoder, or Models which return TorchTensor
            as net_output.
            We need to make a change to support all FairseqEncoder models.
        """
        net_output = model(**sample["net_input"])

        target = model.get_targets(sample, net_output)
        lprobs, loss = self.compute_loss(
            model, net_output, target, reduction, log_probs
        )
        sample_size, logging_output = self.get_logging_output(
            sample, target, lprobs, loss
        )
        alpha = net_output[1]['alpha']
        source_padding_mask = net_output[1]['encoder_padding_mask']
        source_padding_mask = (
            source_padding_mask.t() 
            if source_padding_mask is not None 
            else source_padding_mask 
        )
        targt_padding_mask = target.eq(self.padding_idx)

        latency_loss = self.latency_train.loss(
            alpha,
            source_padding_mask,
            targt_padding_mask
        )

        loss += latency_loss

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "nframes": nframes,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output
