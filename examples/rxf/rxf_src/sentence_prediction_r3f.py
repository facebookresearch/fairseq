# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("sentence_prediction_r3f")
class SentencePredictionR3F(FairseqCriterion):
    def __init__(
        self,
        task,
        eps,
        r3f_lambda,
        noise_type,
        classification_head_name,
        regression_target,
    ):
        super().__init__(task)
        self.eps = eps
        self.r3f_lambda = r3f_lambda
        self.noise_type = noise_type
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target
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
        # fmt: off
        parser.add_argument('--eps', type=float, default=1e-5,
                            help='noise eps')
        parser.add_argument('--r3f-lambda', type=float, default=1.0,
                            help='lambda for combining logistic loss and noisy KL loss')
        parser.add_argument('--noise-type', type=str, default='uniform',
                            choices=['normal', 'uniform'],
                            help='type of noises for RXF methods')
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
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
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        token_embeddings = model.encoder.sentence_encoder.embed_tokens(
            sample["net_input"]["src_tokens"]
        )
        input_logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
            token_embeddings=token_embeddings,
        )
        if model.training and self.noise_sampler:
            noise = self.noise_sampler.sample(sample_shape=token_embeddings.shape).to(
                token_embeddings
            )
            noised_embeddings = token_embeddings.detach().clone() + noise

            noised_logits, _ = model(
                **sample["net_input"],
                features_only=True,
                classification_head_name=self.classification_head_name,
                token_embeddings=noised_embeddings,
            )
            symm_kl = self._get_symm_kl(noised_logits, input_logits)
        else:
            symm_kl = 0

        targets = model.get_targets(sample, [input_logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            loss = F.nll_loss(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                targets,
                reduction="sum",
            )
            if model.training:
                symm_kl = symm_kl * sample_size
                loss = loss + self.r3f_lambda * symm_kl
        else:
            logits = input_logits.squeeze().float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        if not self.regression_target:
            preds = input_logits.max(dim=1)[1]
            logging_output.update(ncorrect=(preds == targets).sum().item())

            if model.training and self.noise_sampler:
                logging_output.update(
                    symm_kl=utils.item(symm_kl.data) if reduce else symm_kl.data
                )
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        symm_kl_sum = sum(log.get("symm_kl", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "symm_kl": symm_kl_sum / sample_size,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect / nsentences)

        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output
