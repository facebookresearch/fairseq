# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class MultiSentencePredictionConfig(FairseqDataclass):
    ignore_str: Optional[str] = field(default="",  metadata={"help": "str to ignore"})
    classification_head_name: str = field(
        default="sentence_classification_head",
        metadata={"help": "name of the classification head to use"},
    )
    number_of_targets: int = field(default=-1,  metadata={"help": "number of the classification heads to use"})


@register_criterion("multi_sentence_prediction", dataclass=MultiSentencePredictionConfig)
class SentencePredictionCriterion(FairseqCriterion):
    def __init__(self, cfg: MultiSentencePredictionConfig, task):
        super().__init__(task)
        self.classification_head_name: str = cfg.classification_head_name
        self.number_of_targets: int = cfg.number_of_targets

        self.ignore_index: int = (self.task.label_dictionary.index(cfg.ignore_str.strip()) if (
            cfg.ignore_str is not None and len(cfg.ignore_str) > 0) else -100)
        print(f"Ignoring Index {self.ignore_index} for string {cfg.ignore_str}")
        assert self.number_of_targets > 0

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        aggregate_loss = 0
        logging_output = {}
        sample_size = None
        for target_index in range(self.number_of_targets):
            classification_head = f"{self.classification_head_name}{target_index}"
            assert (
                hasattr(model, "classification_heads")
                and classification_head in model.classification_heads
            ), f"model must provide correct head not {classification_head}"

            logits, _ = model(
                **sample["net_input"],
                features_only=True,
                classification_head_name=classification_head,
            )
            targets = sample["target"][str(target_index)].view(-1)
            sample_size = targets.numel()

            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            task_loss = F.nll_loss(lprobs, targets, reduction="sum", ignore_index=self.ignore_index)

            loss = task_loss
            # mha & ffn regularization update
            if (
                hasattr(model.args, "mha_reg_scale_factor")
                and model.args.mha_reg_scale_factor != 0.0
            ):
                mha_reg_loss = model._get_adaptive_head_loss()
                loss += mha_reg_loss
                logging_output.update({"mha_reg_loss": mha_reg_loss})
            if (
                hasattr(model.args, "ffn_reg_scale_factor")
                and model.args.ffn_reg_scale_factor != 0.0
            ):
                ffn_reg_loss = model._get_adaptive_ffn_loss()
                loss += ffn_reg_loss
                logging_output.update({"ffn_reg_loss": ffn_reg_loss})

            aggregate_loss += loss

            preds = logits.argmax(dim=1)
            logging_output[f"ncorrect_{target_index}"] = (preds == targets).sum()

        logging_output.update(
            {
                "loss": aggregate_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
            })
        return aggregate_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mha_reg_loss_sum = sum(log.get("mha_reg_loss", 0) for log in logging_outputs)
        ffn_reg_loss_sum = sum(log.get("ffn_reg_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if mha_reg_loss_sum:
            metrics.log_scalar(
                "mha_reg_loss",
                mha_reg_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if ffn_reg_loss_sum:
            metrics.log_scalar(
                "ffn_reg_loss",
                ffn_reg_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0:
            current_index = 0
            current_correct = f"ncorrect_{current_index}"
            while current_correct in logging_outputs[0]:
                ncorrect = sum(log.get(current_correct, 0) for log in logging_outputs)
                metrics.log_scalar(
                    f"accuracy_{current_index}", 100.0 * ncorrect / nsentences, nsentences, round=1
                )
                current_index += 1
                current_correct = f"ncorrect_{current_index}"

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
