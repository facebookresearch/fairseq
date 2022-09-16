# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from collections import OrderedDict

import torch

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterion
from fairseq.criterions.label_smoothed_cross_entropy_with_rdrop import (
    RdropLabelSmoothedCrossEntropyCriterion,
    RdropLabelSmoothedCrossEntropyCriterionConfig,
    duplicate_input,
)
from fairseq.criterions.tacotron2_loss import (
    Tacotron2Criterion,
    Tacotron2CriterionConfig,
)

logger = logging.getLogger(__name__)


class MultitaskCriterion:
    def __init__(self, multitask_tasks, rdrop_alpha=0.0):
        self.rdrop_alpha = rdrop_alpha
        self.rdrop_alpha_mtl = rdrop_alpha

        self.multitask_criterion = OrderedDict()
        self.multitask_loss_weight = OrderedDict()
        for task_name, task_obj in multitask_tasks.items():
            rdrop_alpha_task = task_obj.args.rdrop_alpha
            if rdrop_alpha_task is None:
                rdrop_alpha_task = rdrop_alpha
            self.rdrop_alpha_mtl = rdrop_alpha_task
            logger.info(f"rdrop_alpha is set to {rdrop_alpha_task}")

            if task_obj.args.decoder_type == "ctc":
                self.multitask_criterion[task_name] = CtcCriterion(
                    task_obj.args.criterion_cfg,
                    task_obj,
                    rdrop_alpha=rdrop_alpha_task,
                )
            else:
                self.multitask_criterion[
                    task_name
                ] = RdropLabelSmoothedCrossEntropyCriterion(
                    task_obj,
                    task_obj.args.criterion_cfg.sentence_avg,
                    label_smoothing=task_obj.args.criterion_cfg.label_smoothing,
                    rdrop_alpha=rdrop_alpha_task,
                )

    def set_multitask_loss_weight(self, task_name, weight=0.0):
        self.multitask_loss_weight[task_name] = weight

    def get_multitask_loss(self, model, sample, model_out):
        logging_output = {}
        loss = 0.0
        for task_name, task_criterion in self.multitask_criterion.items():
            layer_id = task_criterion.task.args.input_layer
            if isinstance(task_criterion, CtcCriterion):
                if task_criterion.task.args.input_from == "encoder":
                    if len(model_out["encoder_padding_mask"]) > 0:
                        non_padding_mask = ~model_out["encoder_padding_mask"][0]
                        input_lengths = non_padding_mask.long().sum(-1)
                    else:
                        out = model_out["encoder_states"][layer_id]
                        input_lengths = out.new_full(
                            (out.shape[1],), out.shape[0]
                        ).long()

                    task_sample = {
                        "net_input": {
                            "src_tokens": model_out["encoder_states"][
                                layer_id
                            ],  # check batch idx
                            "src_lengths": input_lengths,
                        },
                        "id": sample["id"],
                    }
                else:
                    task_sample = {
                        "net_input": {
                            "src_tokens": model_out["inner_states"][layer_id],
                            "src_lengths": sample["target_lengths"],
                        },
                        "id": sample["id"],
                    }
            else:
                task_sample = {
                    "net_input": {
                        "src_tokens": sample["multitask"][task_name]["net_input"][
                            "prev_output_tokens"
                        ],
                        "encoder_out": {
                            "encoder_out": [model_out["encoder_states"][layer_id]],
                            "encoder_padding_mask": model_out["encoder_padding_mask"],
                        },
                    }
                }

            for key in ["target", "target_lengths", "ntokens"]:
                task_sample[key] = sample["multitask"][task_name][key]

            task_loss, task_sample_size, task_logging_output = task_criterion(
                model.multitask_decoders[task_name], task_sample
            )

            loss = loss + self.multitask_loss_weight[task_name] * task_loss
            task_logging_output["loss_weight"] = self.multitask_loss_weight[task_name]
            logging_output[task_name] = task_logging_output
        return loss, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        for task_name in logging_outputs[0]["multitask"].keys():
            # different criterion may return different logging
            # currently only reduce on loss, the most common one
            # ideally the way that losses are reduced should also depend on the task type
            loss_sum = sum(
                log["multitask"][task_name].get("loss", 0) for log in logging_outputs
            )
            sample_size = sum(
                log["multitask"][task_name].get("sample_size", 0)
                for log in logging_outputs
            )

            metrics.log_scalar(
                f"multitask_{task_name}_loss",
                loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

            loss_weight = logging_outputs[0]["multitask"][task_name].get(
                "loss_weight", 0
            )
            metrics.log_scalar(
                f"multitask_{task_name}_loss_weight",
                loss_weight,
                weight=0,
                priority=250,
            )


@register_criterion(
    "speech_to_unit", dataclass=RdropLabelSmoothedCrossEntropyCriterionConfig
)
class SpeechToUnitMultitaskTaskCriterion(
    RdropLabelSmoothedCrossEntropyCriterion, MultitaskCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        rdrop_alpha=0.0,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            rdrop_alpha,
        )
        MultitaskCriterion.__init__(self, task.multitask_tasks, rdrop_alpha)

    def forward(self, model, sample, reduce=True):
        net_input_concat = {
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "tgt_speaker": sample["net_input"].get("tgt_speaker", None),
            "return_all_hiddens": True,
        }

        if self.rdrop_alpha > 0 or self.rdrop_alpha_mtl > 0:
            net_input_concat = duplicate_input(net_input_concat)

        net_output, extra = model(**net_input_concat)
        loss, nll_loss, rdrop_kl_loss = self.compute_loss(
            model, [net_output], sample, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, [net_output], sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        if self.rdrop_alpha > 0:
            logging_output["rdrop_kl_loss"] = utils.item(rdrop_kl_loss.data)

        if len(self.multitask_criterion) == 0:
            return loss, sample_size, logging_output

        # multitask
        multitask_loss, multitask_log = self.get_multitask_loss(model, sample, extra)
        loss += multitask_loss
        logging_output["multitask"] = multitask_log

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        # inference metrics
        if "targ_frames" in logging_outputs[0]:
            n = sum(log.get("norm_frames", 0) for log in logging_outputs)
            for key, new_key in [
                ("mcd_loss", "mcd_loss"),
                ("pred_frames", "pred_ratio"),
                ("nins", "ins_rate"),
                ("ndel", "del_rate"),
            ]:
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(new_key, val / n, n, round=3)

        if "multitask" not in logging_outputs[0]:
            return

        MultitaskCriterion.reduce_metrics(logging_outputs)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


@register_criterion("speech_to_spectrogram", dataclass=Tacotron2CriterionConfig)
class SpeechToSpectrogramMultitaskTaskCriterion(Tacotron2Criterion, MultitaskCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        use_guided_attention_loss,
        guided_attention_loss_sigma,
        bce_pos_weight,
        ctc_weight,
    ):
        super().__init__(
            task,
            sentence_avg,
            use_guided_attention_loss,
            guided_attention_loss_sigma,
            bce_pos_weight,
            ctc_weight,
        )
        MultitaskCriterion.__init__(self, task.multitask_tasks)

    def forward(self, model, sample, reduction="mean"):
        bsz, max_len, _ = sample["target"].size()
        feat_tgt = sample["target"]
        feat_len = sample["target_lengths"].view(bsz, 1).expand(-1, max_len)
        eos_tgt = torch.arange(max_len).to(sample["target"].device)
        eos_tgt = eos_tgt.view(1, max_len).expand(bsz, -1)
        eos_tgt = (eos_tgt == (feat_len - 1)).float()

        feat_out, eos_out, extra = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            tgt_speaker=sample["net_input"]["tgt_speaker"],
            target_lengths=sample["target_lengths"],
            return_all_hiddens=True,
        )

        l1_loss, mse_loss, eos_loss = self.compute_loss(
            extra["feature_out"],
            feat_out,
            eos_out,
            feat_tgt,
            eos_tgt,
            sample["target_lengths"],
            reduction,
        )
        attn_loss = torch.tensor(0.0).type_as(l1_loss)
        if self.guided_attn is not None:
            attn_loss = self.guided_attn(
                extra["attn"],
                sample["net_input"]["src_lengths"],
                sample["target_lengths"],
                reduction,
            )
        loss = (
            l1_loss + mse_loss + eos_loss + attn_loss
        )  # do not include ctc loss as there's no text target

        sample_size = sample["nsentences"] if self.sentence_avg else sample["ntokens"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "l1_loss": utils.item(l1_loss.data),
            "mse_loss": utils.item(mse_loss.data),
            "eos_loss": utils.item(eos_loss.data),
            "attn_loss": utils.item(attn_loss.data),
        }

        if len(self.multitask_criterion) == 0:
            return loss, sample_size, logging_output

        # multitask
        multitask_loss, multitask_log = self.get_multitask_loss(model, sample, extra)
        loss += multitask_loss
        logging_output["multitask"] = multitask_log
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        # inference metrics
        if "targ_frames" in logging_outputs[0]:
            n = sum(log.get("norm_frames", 0) for log in logging_outputs)
            for key, new_key in [
                ("mcd_loss", "mcd_loss"),
                ("pred_frames", "pred_ratio"),
                ("nins", "ins_rate"),
                ("ndel", "del_rate"),
            ]:
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(new_key, val / n, n, round=3)

        if "multitask" not in logging_outputs[0]:
            return

        MultitaskCriterion.reduce_metrics(logging_outputs)
