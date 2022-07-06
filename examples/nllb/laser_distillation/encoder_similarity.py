# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn import MSELoss

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("encoder_similarity")
class EncoderSimilarityCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, contrast, margin):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.loss = torch.nn.CosineEmbeddingLoss(margin=margin, reduction="sum")
        self.mse_loss = MSELoss()
        self.contrast = contrast

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--contrast', default=False, action='store_true',
                            help='whether we apply negative sampling')

        parser.add_argument('--margin', default=0., type=float,
                            help='cosine margin for negative sampling')
        # fmt: on

    def forward(self, model, sample, teacher_order, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_input = sample["net_input"]
        net_output = model.encoder.forward(
            net_input["src_tokens"], net_input["src_lengths"], ""
        )

        student_enc_out = net_output["sentemb"]
        student_enc_out = student_enc_out[teacher_order]

        teacher_enc_out = sample["teacher_enc_out"]["sentemb"]

        if self.contrast:
            """
            create (batch_size ** 2) examples. positive examples occur when i % batch_size == 0
            and negative otherwise
            """
            bsz = student_enc_out.size(0)
            indices = torch.arange(bsz)
            student_indices = indices.repeat_interleave(bsz)
            teacher_indices = indices.repeat(bsz)
            y_eq = student_indices == teacher_indices

            y = (y_eq.int() - (1 - y_eq.int())).to(student_enc_out.get_device())

            student = student_enc_out[student_indices]
            teacher = teacher_enc_out[teacher_indices]
        else:
            teacher = teacher_enc_out
            student = student_enc_out
            y = torch.ones(student_enc_out.size(0), device=student_enc_out.device)

        loss = self.loss(student, teacher, y)

        sample_size = teacher_enc_out.size(0)
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": teacher_enc_out.size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens, ntokens, round=3)
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
