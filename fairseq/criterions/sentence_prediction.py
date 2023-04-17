# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as _matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def matthews_corrcoef(preds, labels):
    # make it consistent with other metrics taking (preds, labels) as input
    mcc = _matthews_corrcoef(labels, preds)
    return mcc


@dataclass
class SentencePredictionConfig(FairseqDataclass):
    classification_head_name: str = field(
        default="sentence_classification_head",
        metadata={"help": "name of the classification head to use"},
    )
    regression_target: bool = field(
        default=False,
    )
    report_mcc: bool = False
    report_acc_and_f1: bool = False
    report_pearson_and_spearman: bool = False


@register_criterion("sentence_prediction", dataclass=SentencePredictionConfig)
class SentencePredictionCriterion(FairseqCriterion):
    def __init__(self, cfg: SentencePredictionConfig, task):
        super().__init__(task)
        self.classification_head_name = cfg.classification_head_name
        self.regression_target = cfg.regression_target
        self.keep_pred_and_targ = (
            cfg.report_mcc or cfg.report_acc_and_f1 or cfg.report_pearson_and_spearman
        )
        self.report_mcc = cfg.report_mcc
        self.report_acc_and_f1 = cfg.report_acc_and_f1
        self.report_pearson_and_spearman = cfg.report_pearson_and_spearman
        self.label_dict = task.label_dictionary

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

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            task_loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            task_loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {}
        loss = task_loss
        # mha & ffn regularization update
        if (
            hasattr(model, "args")
            and hasattr(model.args, "mha_reg_scale_factor")
            and model.args.mha_reg_scale_factor != 0.0
        ):
            mha_reg_loss = model._get_adaptive_head_loss()
            loss += mha_reg_loss
            logging_output.update({"mha_reg_loss": mha_reg_loss})
        if (
            hasattr(model, "args")
            and hasattr(model.args, "ffn_reg_scale_factor")
            and model.args.ffn_reg_scale_factor != 0.0
        ):
            ffn_reg_loss = model._get_adaptive_ffn_loss()
            loss += ffn_reg_loss
            logging_output.update({"ffn_reg_loss": ffn_reg_loss})

        logging_output.update(
            {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
            }
        )
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()
        if self.keep_pred_and_targ and not model.training:
            if self.regression_target:
                logging_output["pred"] = logits.detach().cpu().tolist()
                logging_output["targ"] = targets.detach().cpu().tolist()
            else:
                # remove offset `self.label_dict.nspecial` from OffsetTokensDataset
                preds = self.label_dict.string(preds + self.label_dict.nspecial).split()
                targets = self.label_dict.string(
                    targets + self.label_dict.nspecial
                ).split()
                logging_output["pred"] = list(map(int, preds))
                logging_output["targ"] = list(map(int, targets))

            if self.report_mcc:
                logging_output["report_mcc"] = True
            if self.report_acc_and_f1:
                logging_output["report_acc_and_f1"] = True
            if self.report_pearson_and_spearman:
                logging_output["report_pearson_and_spearman"] = True

        return loss, sample_size, logging_output

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

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

        # Metrics used by GLUE
        pred = np.array(
            list(chain.from_iterable(log.get("pred", []) for log in logging_outputs))
        )
        targ = np.array(
            list(chain.from_iterable(log.get("targ", []) for log in logging_outputs))
        )
        if len(pred):
            metrics.log_concat_tensor("pred", torch.from_numpy(pred), dim=0)
            metrics.log_concat_tensor("targ", torch.from_numpy(targ), dim=0)
            if any("report_mcc" in log for log in logging_outputs):
                metrics.log_derived(
                    "mcc",
                    lambda meters: safe_round(
                        matthews_corrcoef(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )
                        * 100,
                        1,
                    ),
                )
            if any("report_acc_and_f1" in log for log in logging_outputs):
                metrics.log_derived(
                    "acc_and_f1",
                    lambda meters: safe_round(
                        acc_and_f1(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["acc_and_f1"]
                        * 100,
                        1,
                    ),
                )
                metrics.log_derived(
                    "f1",
                    lambda meters: safe_round(
                        acc_and_f1(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["f1"]
                        * 100,
                        1,
                    ),
                )
            if any("report_pearson_and_spearman" in log for log in logging_outputs):
                metrics.log_derived(
                    "pearson_and_spearman",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["corr"]
                        * 100,
                        1,
                    ),
                )
                metrics.log_derived(
                    "pearson",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["pearson"]
                        * 100,
                        1,
                    ),
                )
                metrics.log_derived(
                    "spearman",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["spearmanr"]
                        * 100,
                        1,
                    ),
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
