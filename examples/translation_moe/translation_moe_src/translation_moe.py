# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from .logsumexp_moe import LogSumExpMoE
from .mean_pool_gating_network import MeanPoolGatingNetwork


METHOD_CHOICES = ChoiceEnum(["sMoElp", "sMoEup", "hMoElp", "hMoEup"])


@dataclass
class TranslationMoEConfig(TranslationConfig):
    method: METHOD_CHOICES = field(
        default="hMoEup",
        metadata={"help": "MoE method"},
    )
    num_experts: int = field(
        default=3,
        metadata={"help": "number of experts"},
    )
    mean_pool_gating_network: bool = field(
        default=False,
        metadata={"help": "use a simple mean-pooling gating network"},
    )
    mean_pool_gating_network_dropout: float = field(
        default=0,
        metadata={"help": "dropout for mean-pooling gating network"},
    )
    mean_pool_gating_network_encoder_dim: int = field(
        default=0,
        metadata={"help": "encoder output dim for mean-pooling gating network"},
    )
    gen_expert: int = field(
        default=0,
        metadata={"help": "which expert to use for generation"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_task("translation_moe", dataclass=TranslationMoEConfig)
class TranslationMoETask(TranslationTask):
    """
    Translation task for Mixture of Experts (MoE) models.

    See `"Mixture Models for Diverse Machine Translation: Tricks of the Trade"
    (Shen et al., 2019) <https://arxiv.org/abs/1902.07816>`_.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    cfg: TranslationMoEConfig

    def __init__(self, cfg: TranslationMoEConfig, src_dict, tgt_dict):
        if cfg.method == "sMoElp":
            # soft MoE with learned prior
            self.uniform_prior = False
            self.hard_selection = False
        elif cfg.method == "sMoEup":
            # soft MoE with uniform prior
            self.uniform_prior = True
            self.hard_selection = False
        elif cfg.method == "hMoElp":
            # hard MoE with learned prior
            self.uniform_prior = False
            self.hard_selection = True
        elif cfg.method == "hMoEup":
            # hard MoE with uniform prior
            self.uniform_prior = True
            self.hard_selection = True

        # add indicator tokens for each expert
        for i in range(cfg.num_experts):
            # add to both dictionaries in case we're sharing embeddings
            src_dict.add_symbol("<expert_{}>".format(i))
            tgt_dict.add_symbol("<expert_{}>".format(i))

        super().__init__(cfg, src_dict, tgt_dict)

    def build_model(self, cfg, from_checkpoint=False):
        from fairseq import models

        model = models.build_model(cfg, self)
        if not self.uniform_prior and not hasattr(model, "gating_network"):
            if self.cfg.mean_pool_gating_network:
                if self.cfg.mean_pool_gating_network_encoder_dim > 0:
                    encoder_dim = self.cfg.mean_pool_gating_network_encoder_dim
                elif getattr(cfg, "encoder_embed_dim", None):
                    # assume that encoder_embed_dim is the encoder's output dimension
                    encoder_dim = cfg.encoder_embed_dim
                else:
                    raise ValueError(
                        "Must specify --mean-pool-gating-network-encoder-dim"
                    )

                if self.cfg.mean_pool_gating_network_dropout > 0:
                    dropout = self.cfg.mean_pool_gating_network_dropout
                elif getattr(cfg, "dropout", None):
                    dropout = cfg.dropout
                else:
                    raise ValueError("Must specify task.mean_pool_gating_network_dropout")

                model.gating_network = MeanPoolGatingNetwork(
                    encoder_dim,
                    self.cfg.num_experts,
                    dropout,
                )
            else:
                raise ValueError(
                    "translation_moe task with learned prior requires the model to "
                    "have a gating network; try using --mean-pool-gating-network"
                )
        return model

    def expert_index(self, i):
        return i + self.tgt_dict.index("<expert_0>")

    def _get_loss(self, sample, model, criterion):
        assert hasattr(
            criterion, "compute_loss"
        ), "translation_moe task requires the criterion to implement the compute_loss() method"

        k = self.cfg.num_experts
        bsz = sample["target"].size(0)

        def get_lprob_y(encoder_out, prev_output_tokens_k):
            net_output = model.decoder(
                prev_output_tokens=prev_output_tokens_k,
                encoder_out=encoder_out,
            )
            loss, _ = criterion.compute_loss(model, net_output, sample, reduce=False)
            loss = loss.view(bsz, -1)
            return -loss.sum(dim=1, keepdim=True)  # -> B x 1

        def get_lprob_yz(winners=None):
            encoder_out = model.encoder(
                src_tokens=sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
            )

            if winners is None:
                lprob_y = []
                for i in range(k):
                    prev_output_tokens_k = sample["net_input"][
                        "prev_output_tokens"
                    ].clone()
                    assert not prev_output_tokens_k.requires_grad
                    prev_output_tokens_k[:, 0] = self.expert_index(i)
                    lprob_y.append(get_lprob_y(encoder_out, prev_output_tokens_k))
                lprob_y = torch.cat(lprob_y, dim=1)  # -> B x K
            else:
                prev_output_tokens_k = sample["net_input"]["prev_output_tokens"].clone()
                prev_output_tokens_k[:, 0] = self.expert_index(winners)
                lprob_y = get_lprob_y(encoder_out, prev_output_tokens_k)  # -> B

            if self.uniform_prior:
                lprob_yz = lprob_y
            else:
                lprob_z = model.gating_network(encoder_out)  # B x K
                if winners is not None:
                    lprob_z = lprob_z.gather(dim=1, index=winners.unsqueeze(-1))
                lprob_yz = lprob_y + lprob_z.type_as(lprob_y)  # B x K

            return lprob_yz

        # compute responsibilities without dropout
        with utils.model_eval(model):  # disable dropout
            with torch.no_grad():  # disable autograd
                lprob_yz = get_lprob_yz()  # B x K
                prob_z_xy = torch.nn.functional.softmax(lprob_yz, dim=1)
        assert not prob_z_xy.requires_grad

        # compute loss with dropout
        if self.hard_selection:
            winners = prob_z_xy.max(dim=1)[1]
            loss = -get_lprob_yz(winners)
        else:
            lprob_yz = get_lprob_yz()  # B x K
            loss = -LogSumExpMoE.apply(lprob_yz, prob_z_xy, 1)

        loss = loss.sum()
        sample_size = (
            sample["target"].size(0) if self.cfg.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": bsz,
            "sample_size": sample_size,
            "posterior": prob_z_xy.float().sum(dim=0).cpu(),
        }
        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        return loss, sample_size, logging_output

    def inference_step(
        self,
        generator,
        models,
        sample,
        prefix_tokens=None,
        expert=None,
        constraints=None,
    ):
        expert = expert or self.cfg.gen_expert
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=self.expert_index(expert),
            )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        metrics.log_scalar(
            "posterior",
            sum(log["posterior"] for log in logging_outputs if "posterior" in log),
        )
