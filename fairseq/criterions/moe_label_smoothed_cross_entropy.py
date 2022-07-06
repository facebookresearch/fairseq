# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from fairseq import distributed_utils, metrics, utils
from fairseq.criterions import FairseqCriterion, MoECriterionConfig, register_criterion
from fairseq.logging.meters import GroupedAverageMeter
from fairseq.modules.moe import MOELayer


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@dataclass
class MoELabelSmoothedCrossEntropyCriterionConfig(MoECriterionConfig):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: Optional[bool] = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: Optional[int] = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    moe_gate_loss_wt: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Weight associated with MoE gate loss in the weighted sum of gate loss and cross entropy loss"
        },
    )
    moe_gate_loss_combine_method: Optional[str] = field(
        default="average",
        metadata={
            "help": "Method of combining the gate loss from each MoE layers ('sum', 'average')"
        },
    )
    cmr_gate_loss_p: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "In CMR loss, encourages fraction of tokens p to be routed to MOE layers"
        },
    )
    cmr_gate_loss_wt: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Weight associated with CMR loss in the weighted sum of CMR, MoE gate loss, and cross entropy loss"
        },
    )
    moe_cmr_xgpu: Optional[bool] = field(
        default=False,
        metadata={"help": "All2all across gpus when computing CMR gate loss"},
    )


@register_criterion(
    "moe_label_smoothed_cross_entropy",
    dataclass=MoELabelSmoothedCrossEntropyCriterionConfig,
)
class MoELabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    moe_logging_keys = [
        "overflow_expert1",  # average % of overflowed tokens from 1st expert
        "overflow_expert2",  # average % of overflowed tokens from 2nd expert
        "entropy_gating",  # average entropy of the gating distribution
        "expert1_balance_top",  # average cumulative % of tokens processed by the most used 20% 1st experts
        "expert1_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 1st experts
        "unused_expert1_count",  # average number of 1st experts which process no tokens
        "expert2_balance_top",  # average cumulative % of tokens processed by the most used 20% 2nd experts
        "expert2_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 2nd experts
        "unused_expert2_count",  # average number of 2nd experts which process no tokens
        "all_to_all_cpu_time_ms",  # CPU time spent in all to all calls in milliseconds
        "all_to_all_cuda_time_ms",  # CUDA ttime spent in all to all calls in milliseconds
        "median_prefix_count_expert1",
        "cmr_lang_gates",
    ]
    secondary_moe_logging_keys = [
        "median_prefix_count_expert1_encoder",
        "median_prefix_count_expert1_decoder",
        "median_prefix_count_expert1_encoder_1st_layer",
        "median_prefix_count_expert1_encoder_last_layer",
        "median_prefix_count_expert1_decoder_1st_layer",
        "median_prefix_count_expert1_decoder_last_layer",
    ]

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        moe_gate_loss_wt,
        moe_gate_loss_combine_method,
        ignore_prefix_size=0,
        report_accuracy=False,
        moe_cmr_xgpu=False,
        cmr_gate_loss_p=1.0,
        cmr_gate_loss_wt=0.0,
    ):
        super().__init__(task)
        self.task = task
        self.source_dictionary = getattr(task, "source_dictionary", None)
        self.lang_idx = getattr(task, "lang_idx", None)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.gate_loss_weight = moe_gate_loss_wt
        self.gate_loss_combine_method = moe_gate_loss_combine_method
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.moe_cmr_xgpu = moe_cmr_xgpu
        self.cmr_gate_loss_p = cmr_gate_loss_p
        self.cmr_gate_loss_weight = cmr_gate_loss_wt

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        loss, nll_loss, moe_loss, cmr_loss, moe_metadata = self.compute_loss(
            model, net_output, sample, sample_size, reduce=reduce
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "moe_loss": moe_loss.data,
            "cmr_loss": cmr_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        logging_output.update(moe_metadata)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, sample_size, reduce=True):
        gate_loss = 0.0
        gate_count = 0
        for l_gate_loss in net_output[1]["moe_gate_loss"]:
            if l_gate_loss is not None:
                gate_loss += l_gate_loss
                gate_count += 1

        # avoid float16 overflow
        cmr_gate_used_sum = torch.zeros_like(gate_loss, dtype=torch.float32)
        cmr_gate_total_sum = torch.zeros_like(gate_loss, dtype=torch.float32)
        for cmr_gate_used in net_output[1]["cmr_gate_loss_num"]:
            if cmr_gate_used is not None:
                cmr_gate_used_sum += cmr_gate_used
        for cmr_gate_total in net_output[1]["cmr_gate_loss_denom"]:
            if cmr_gate_total is not None:
                cmr_gate_total_sum += cmr_gate_total
        if self.moe_cmr_xgpu:
            cmr_gate_used_sum = distributed_utils.all_reduce(
                cmr_gate_used_sum,
                group=distributed_utils.get_data_parallel_group(),
                op="sum",
            )
            cmr_gate_total_sum = distributed_utils.all_reduce(
                cmr_gate_total_sum,
                group=distributed_utils.get_data_parallel_group(),
                op="sum",
            )

        cmr_gate_loss = torch.zeros_like(gate_loss)
        if self.cmr_gate_loss_weight > 0:
            cmr_gate_loss = (
                (cmr_gate_used_sum / cmr_gate_total_sum.clamp(1e-5))
                - self.cmr_gate_loss_p
            ).abs()
        if self.gate_loss_combine_method == "average":
            gate_loss = gate_loss / gate_count
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        ls_loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        gate_loss = sample_size * gate_loss
        cmr_gate_loss = sample_size * cmr_gate_loss
        loss = (
            ls_loss
            + self.gate_loss_weight * gate_loss
            + self.cmr_gate_loss_weight * cmr_gate_loss
        )
        return loss, nll_loss, gate_loss, cmr_gate_loss, self.get_moe_metadata(model)

    def get_moe_metadata(self, model):
        moe_logging_output = {}
        for key in MoELabelSmoothedCrossEntropyCriterion.moe_logging_keys:
            vals = []
            module_names = []
            for name, module in model.named_modules():
                if isinstance(module, MOELayer):
                    val = module.metadata[key] if key in module.metadata else 0
                    vals.append(val)
                    module_names.append(name)
            if key == "median_prefix_count_expert1":
                encoder_vals = [
                    v for name, v in zip(module_names, vals) if ".encoder." in name
                ]
                moe_logging_output["median_prefix_count_expert1_encoder"] = sum(
                    encoder_vals
                ) / len(encoder_vals)
                moe_logging_output[
                    "median_prefix_count_expert1_encoder_1st_layer"
                ] = encoder_vals[0]
                moe_logging_output[
                    "median_prefix_count_expert1_encoder_last_layer"
                ] = encoder_vals[-1]
                decoder_vals = [
                    v for name, v in zip(module_names, vals) if ".decoder." in name
                ]
                moe_logging_output["median_prefix_count_expert1_decoder"] = sum(
                    decoder_vals
                ) / len(decoder_vals)
                moe_logging_output[
                    "median_prefix_count_expert1_decoder_1st_layer"
                ] = decoder_vals[0]
                moe_logging_output[
                    "median_prefix_count_expert1_decoder_last_layer"
                ] = decoder_vals[-1]
            moe_logging_output[key] = sum(vals) / len(vals)
        moe_logging_output["batch_count"] = 1
        return moe_logging_output

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        moe_loss_sum = sum(log.get("moe_loss", 0) for log in logging_outputs)
        # TODO: CMR loss doesn't make sense during validation bc examples aren't random
        cmr_loss_sum = sum(log.get("cmr_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "moe_gate_loss", moe_loss_sum / sample_size, sample_size, round=8
        )
        metrics.log_scalar(
            "cmr_gate_loss", cmr_loss_sum / sample_size, sample_size, round=8
        )
        batch_count = sum(log.get("batch_count", 0) for log in logging_outputs)

        cmr_lang_gates = sum(log.get("cmr_lang_gates", 0) for log in logging_outputs)
        if torch.is_tensor(cmr_lang_gates) and torch.numel(cmr_lang_gates) > 1:
            lang_idx = None
            for log in logging_outputs:
                if log.get("lang_idx", None) is not None:
                    lang_idx = log["lang_idx"]
                    break
            if lang_idx is None:
                raise ValueError("logging outputs should contain lang_idx")
            metrics.log_custom(
                lambda: GroupedAverageMeter(["NONE"] + lang_idx, round=8),
                "cmr_lang_gates_d",
                cmr_lang_gates / batch_count,
                batch_count,
            )

        all_keys = (
            MoELabelSmoothedCrossEntropyCriterion.moe_logging_keys
            + MoELabelSmoothedCrossEntropyCriterion.secondary_moe_logging_keys
        )
        for key in all_keys:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, val / batch_count, batch_count, round=3)
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
