# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from omegaconf import II
from torch.nn.modules.loss import _Loss

from fairseq import metrics, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass


class FairseqCriterion(_Loss):
    def __init__(self, task):
        super().__init__()
        self.task = task
        if hasattr(task, "target_dictionary"):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_criterion(cls, cfg: FairseqDataclass, task):
        """Construct a criterion from command-line args."""
        # arguments in the __init__.
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if (
                p.kind == p.POSITIONAL_ONLY
                or p.kind == p.VAR_POSITIONAL
                or p.kind == p.VAR_KEYWORD
            ):
                # we haven't implemented inference for these argument types,
                # but PRs welcome :)
                raise NotImplementedError("{} not supported".format(p.kind))

            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}

            if p.name == "task":
                init_args["task"] = task
            elif p.name == "cfg":
                init_args["cfg"] = cfg
            elif hasattr(cfg, p.name):
                init_args[p.name] = getattr(cfg, p.name)
            elif p.default != p.empty:
                pass  # we'll use the default value
            else:
                raise NotImplementedError(
                    "Unable to infer Criterion arguments, please implement "
                    "{}.build_criterion".format(cls.__name__)
                )
        return cls(**init_args)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(
        logging_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "The aggregate_logging_outputs API is deprecated. "
            "Please use the reduce_metrics API instead."
        )
        raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "Criterions should implement the reduce_metrics API. "
            "Falling back to deprecated aggregate_logging_outputs API."
        )
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {"nsentences", "ntokens", "sample_size"}:
                continue
            metrics.log_scalar(k, v)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


class LegacyFairseqCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task=task)
        self.args = args

        utils.deprecation_warning(
            "Criterions should take explicit arguments instead of an "
            "argparse.Namespace object, please update your criterion by "
            "extending FairseqCriterion instead of LegacyFairseqCriterion."
        )

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)


@dataclass
class MoECriterionConfig(FairseqDataclass):
    moe_gate_loss_wt: float = field(
        default=1.0,
        metadata={
            "help": "Weight associated with MoE gate loss"
            "in the weighted sum of gate loss and cross entropy loss"
        },
    )
    moe_gate_loss_combine_method: str = field(
        default="average",
        metadata={
            "help": "Method of combining the gate loss from each MoE layers"
            "('sum', 'average')"
        },
    )
    moe_gate_loss_transform: str = field(
        default="none",
        metadata={
            "help": "Transformation to apply to the gate loss ('none', 'neg_log')"
        },
    )
    sentence_avg: bool = II("optimization.sentence_avg")


class MoECriterion(FairseqCriterion):

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
    ]

    def __init__(
        self,
        task,
        moe_gate_loss_wt,
        moe_gate_loss_combine_method,
        moe_gate_loss_transform,
        sentence_avg,
    ):
        super().__init__(task)
        self.gate_loss_weight = moe_gate_loss_wt
        self.gate_loss_combine_method = moe_gate_loss_combine_method
        self.gate_loss_transform = moe_gate_loss_transform
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        (
            loss,
            inner_loss,
            moe_loss,
            moe_metadata,
            sample_size,
            logging_output,
        ) = self.compute_loss(model, sample, reduce=reduce)

        logging_output["loss"] = loss.data
        logging_output["moe_loss"] = moe_loss.data
        logging_output.update(moe_metadata)

        return loss, sample_size, logging_output

    def compute_loss(self, model, sample, reduce=True):
        net_output, inner_loss, sample_size, logging_output = self.compute_inner_loss(
            model, sample
        )
        gate_loss = 0.0
        gate_count = 0
        for l_aux in net_output[1]["moe_gate_loss"]:
            if l_aux is not None:
                gate_loss += l_aux
                gate_count += 1
        if self.gate_loss_combine_method == "average":
            gate_loss = gate_loss / gate_count
        if self.gate_loss_transform == "neg_log":
            gate_loss = -torch.log(gate_loss)
        gate_loss = sample_size * gate_loss
        loss = inner_loss + self.gate_loss_weight * gate_loss
        return (
            loss,
            inner_loss,
            gate_loss,
            self.get_moe_metadata(model),
            sample_size,
            logging_output,
        )

    def compute_inner_loss(self, model, sample):
        """Compute the non-MoE portion of the loss. Default is cross-entropy"""
        raise NotImplementedError

    def get_moe_metadata(self, model):
        from fairseq.modules.moe import MOELayer

        moe_logging_output = {}
        for key in MoECriterion.moe_logging_keys:
            total_val = 0
            count = 0
            for _, module in model.named_modules():
                if isinstance(module, MOELayer):
                    total_val += module.metadata[key] if key in module.metadata else 0
                    count += 1
            moe_logging_output[key] = total_val / count
        moe_logging_output["batch_count"] = 1
        return moe_logging_output

    @staticmethod
    def reduce_moe_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        moe_loss_sum = sum(log.get("moe_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "moe_gate_loss", moe_loss_sum / sample_size, sample_size, round=8
        )
        batch_count = sum(log.get("batch_count", 0) for log in logging_outputs)
        for key in MoECriterion.moe_logging_keys:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, val / batch_count, batch_count, round=3)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        raise NotImplementedError

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
