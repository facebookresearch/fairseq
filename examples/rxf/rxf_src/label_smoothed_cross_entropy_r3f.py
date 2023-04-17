# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion("label_smoothed_cross_entropy_r3f")
class LabelSmoothedCrossEntropyR3FCriterion(FairseqCriterion):
    def __init__(
        self, task, sentence_avg, label_smoothing, eps, r3f_lambda, noise_type
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.r3f_lambda = r3f_lambda
        self.noise_type = noise_type
        if self.noise_type in {"normal"}:
            self.noise_sampler = torch.distributions.normal.Normal(
                loc=0.0, scale=self.eps
            )
        elif self.noise_type == "uniform":
            self.noise_sampler = torch.distributions.uniform.Uniform(
                low=-self.eps, high=self.eps
            )
        else:
            raise Exception(f"unrecognized noise type {self.noise_type}")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--eps', type=float, default=1e-5,
                            help='noise eps')
        parser.add_argument('--r3f-lambda', type=float, default=1.0,
                            help='lambda for combining logistic loss and noisy KL loss')
        parser.add_argument('--noise-type', type=str, default='normal',
                            choices=['normal', 'uniform'],
                            help='type of noises')
        # fmt: on

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        ) / noised_logits.size(0)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        token_embeddings = model.encoder.embed_tokens(sample["net_input"]["src_tokens"])
        input_logits, extra = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(
            model, (input_logits, extra), sample, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        if model.training:
            noise = self.noise_sampler.sample(sample_shape=token_embeddings.shape).to(
                token_embeddings
            )
            noised_embeddings = token_embeddings.clone() + noise

            noised_logits, _ = model(
                **sample["net_input"], token_embeddings=noised_embeddings
            )
            symm_kl = self._get_symm_kl(noised_logits, input_logits)

        if model.training:
            symm_kl = symm_kl * sample_size
            loss = loss + self.r3f_lambda * symm_kl

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if model.training:
            logging_output.update(
                symm_kl=utils.item(symm_kl.data) if reduce else symm_kl.data
            )

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.label_smoothing,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        symm_kl_sum = sum(log.get("symm_kl", 0) for log in logging_outputs)

        metrics.log_scalar("symm_kl", symm_kl_sum / sample_size, sample_size, round=3)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
