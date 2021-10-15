# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import MoECriterion, register_criterion, MoECriterionConfig

@dataclass
class MoEMaskedLmCriterionConfig(MoECriterionConfig):
    tpu: bool = field(
        default=False,
        metadata={
            "help": "Whether we are training on TPUs or not"
        }
    )

@register_criterion("moe_masked_lm", dataclass=MoEMaskedLmCriterionConfig)
class MoEMaskedLmLoss(MoECriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, moe_gate_loss_wt, moe_gate_loss_combine_method, moe_gate_loss_transform, sentence_avg, tpu):
        super().__init__(task, moe_gate_loss_wt, moe_gate_loss_combine_method, moe_gate_loss_transform, sentence_avg)
        self.tpu = tpu

    def compute_inner_loss(self, model, sample):
        # compute network mlm loss
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).

        # support TPUs later
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        net_output = model(**sample["net_input"], masked_tokens=masked_tokens)
        logits = net_output[0]
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        logging_output = {
            "inner_loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return net_output, loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        MoEMaskedLmLoss.reduce_moe_metrics(logging_outputs)

        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("inner_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "inner_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["inner_loss"].avg)
        )
