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
class MoESentencePredictionCriterionConfig(MoECriterionConfig):
    classification_head_name: str = field(
        default = "sentence_classification_head",
        metadata={
            "help": "name of the classification head to use"
        }
    )
    regression_target: bool = field(
        default = False,
        metadata={
            "help": "whether target is regression or not"
        }
    )

@register_criterion("moe_sentence_prediction", dataclass=MoESentencePredictionCriterionConfig)
class MoESentencePredictionLoss(MoECriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, moe_gate_loss_wt, moe_gate_loss_combine_method, moe_gate_loss_transform, sentence_avg, classification_head_name, regression_target):
        super().__init__(task, moe_gate_loss_wt, moe_gate_loss_combine_method, moe_gate_loss_transform, sentence_avg)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

    def compute_inner_loss(self, model, sample):
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        logits = net_output[0]
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {
            "inner_loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()
        return net_output, loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        MoESentencePredictionLoss.reduce_moe_metrics(logging_outputs)

        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("inner_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "inner_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
