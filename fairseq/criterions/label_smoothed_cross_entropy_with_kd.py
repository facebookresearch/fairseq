# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import json
from typing import Optional
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from collections import defaultdict

from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class KDLabelSmoothedCrossEntropyCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    kd_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "arguments for knowledge distillation (kd_rate, kd_queue_size, kd_strategy)"
        },
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_kd",
    dataclass=KDLabelSmoothedCrossEntropyCriterionConfig,
)
class KDLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        kd_args=None,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
        )
        self.sentence_avg = sentence_avg
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        # new parameters
        assert kd_args is not None, "Knowledge distillation arguments are missing!"

        kd_args = json.loads(kd_args)

        self.kd_rate = kd_args.get("rate", 0.0)
        self.kd_strategy = kd_args.get(
            "strategy", "word_level"
        )  # Possible values: word_level, word_seq_level, batch_level, global_level, global_language_wise
        self.kd_queue_size = kd_args.get("queue_size", 10000)

        self.num_languages = (
            len(self.task.lang_ids) if self.task.lang_ids is not None else -1
        )

        if self.kd_strategy == "global_language_wise":
            self.queue = {}
            for id in self.task.lang_ids:
                self.queue[id] = torch.tensor([], device=device)
        else:
            self.queue = torch.tensor([], device=device)

    def get_lang_ids(self, tokens):
        non_pad_mask = tokens.ne(self.padding_idx)
        col_indices = torch.max(non_pad_mask, dim=1)[1]
        col_indices = col_indices.unsqueeze(1)
        lang_ids = tokens.gather(1, col_indices)
        return lang_ids.flatten().tolist()

    def push_to_FIFO_queue(self, tensor):
        # this method is applicable only when we have a single global queue
        # here self.queue is torch.cuda.FloatTensor
        tensor = tensor.detach()
        tensor_sz = tensor.size(0)
        current_queue_sz = self.queue.size(0)
        if tensor_sz + current_queue_sz >= self.kd_queue_size:
            self.queue = self.queue[tensor_sz:]
        self.queue = torch.cat((self.queue, tensor))

    def push_to_lang_FIFO_queue(self, id, tensor):
        # this method is applicable only when we have a mulitple global queues
        # here self.queue is dictionary of torch.cuda.FloatTensors
        tensor = tensor.detach()
        tensor_sz = tensor.size(0)
        current_queue_sz = self.queue[id].size(0)
        if tensor_sz + current_queue_sz > self.kd_queue_size:
            self.queue[id] = self.queue[id][tensor_sz:]
        self.queue[id] = torch.cat((self.queue[id], tensor))

    def forward(self, model, teacher_model, sample, update_num=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        assert (
            teacher_model is not None
        ), "knowledge distillation requires a teacher model!"

        # compute teacher outputs
        # make sure to wrap it in a torch.no_grad()
        # as we want the teacher model only on eval mode
        # and not generate any gradients for itself
        with torch.inference_mode():
            teacher_output = teacher_model(**sample["net_input"])
            sample["teacher_output"] = teacher_output

        loss, extra = self.compute_loss(
            model,
            net_output,
            sample,
            teacher_model=teacher_model,
            teacher_output=teacher_output,
        )

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "kd_loss": (
                extra["kd_loss"].data if extra.get("kd_loss", None) is not None else 0
            ),
            "nll_loss": (
                extra["nll_loss"].data
                if extra.get("nll_loss", None) is not None
                else loss.data
            ),
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(
        self, model, net_output, sample, teacher_model=None, teacher_output=None
    ):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        teacher_lprobs, _ = self.get_lprobs_and_target(
            teacher_model, teacher_output, sample
        )

        extra = {}
        pad_mask = target.eq(self.padding_idx)

        # compute preliminary loss and nll_loss of student_model
        golden_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, reduce=False, ignore_index=self.padding_idx
        )

        nll_loss = nll_loss.view(-1)
        golden_loss = golden_loss.view(-1)

        kd_loss = F.kl_div(lprobs, teacher_lprobs, log_target=True, reduction="none")
        kd_loss = kd_loss.sum(dim=-1).masked_fill_(pad_mask, 0.0)

        if self.kd_strategy == "word_level":
            extra["kd_loss"] = kd_loss.sum()
            extra["nll_loss"] = nll_loss.sum()
            loss = extra["kd_loss"]

        elif self.kd_strategy == "word_seq_level":
            extra["kd_loss"] = kd_loss.sum()
            extra["nll_loss"] = nll_loss.sum()
            loss = golden_loss.sum() + extra["kd_loss"]

        elif self.kd_strategy == "batch_level":
            loss_gate = nll_loss.topk(
                math.ceil(nll_loss.size(0) * self.kd_rate), dim=0, largest=True
            )[0][-1]
            KD_mask = nll_loss >= loss_gate
            extra["kd_loss"] = kd_loss[KD_mask].sum()
            extra["nll_loss"] = nll_loss.sum()
            loss = golden_loss.sum() + extra["kd_loss"]

        elif self.kd_strategy == "global_level":
            self.push_to_FIFO_queue(nll_loss)
            loss_gate = self.queue.topk(
                math.ceil(self.queue.size(0) * self.kd_rate), dim=0, largest=True
            )[0][-1]
            KD_mask = nll_loss >= loss_gate  # B * T
            extra["kd_loss"] = kd_loss[KD_mask].sum()
            extra["nll_loss"] = nll_loss.sum()
            loss = golden_loss.sum() + extra["kd_loss"]

        elif self.kd_strategy == "global_language_wise":
            # TODO: Buggy, needs fixing ASAP !!
            indices, kd_loss_ = defaultdict(list), 0
            nll_loss_langwise, kd_loss_langwise = {}, {}
            inp_tokens = sample["net_input"]["src_tokens"]

            for idx, lang_id in enumerate(self.get_lang_ids(inp_tokens)):
                indices[lang_id].append(idx)

            for lang_id, idx in indices.items():
                idx_tensor = torch.cuda.LongTensor(idx)
                nll_loss_lang = nll_loss.index_select(0, idx_tensor).view(-1)
                kd_loss_lang = kd_loss.index_select(0, idx_tensor).view(-1)
                nll_loss_langwise[lang_id] = nll_loss_lang
                kd_loss_langwise[lang_id] = kd_loss_lang
                self.push_to_lang_FIFO_queue(lang_id, nll_loss_lang)

            for lang_id in indices.keys():
                loss_gate = self.queue[lang_id].topk(
                    math.ceil(self.queue[lang_id].size(0) * self.kd_rate),
                    dim=0,
                    largest=True,
                )[0][-1]
                KD_mask = nll_loss_langwise[lang_id] >= loss_gate
                kd_loss_ += kd_loss_langwise[lang_id][KD_mask].sum()

            extra["kd_loss"] = kd_loss_
            extra["nll_loss"] = nll_loss.sum()
            loss = golden_loss.sum() + extra["kd_loss"]

        else:
            raise ValueError("Unknown strategy or parameter mismatch")
        return loss, extra

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # sum metrics
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        kd_loss = sum(log.get("kd_loss", 0) for log in logging_outputs)
        # log metrics
        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar("kd_loss", kd_loss / ntokens / math.log(2), ntokens, round=3)
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
                lambda meters: (
                    round(meters["n_correct"].sum * 100.0 / meters["total"].sum, 3)
                    if meters["total"].sum > 0
                    else float("nan")
                ),
            )
