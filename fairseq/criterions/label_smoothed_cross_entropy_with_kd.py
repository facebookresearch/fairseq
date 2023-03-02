# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from dataclasses import dataclass, field

import torch
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


@dataclass
class KDLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    kd_rate: Optional[float] = field(
        default=None,
        metadata={"help": "the hyperparameter `tau` to control the number of words to get distillation knowledge"}
    )
    kd_queue_size: Optional[int] = field(
        default=20000, 
        metadata={"help": "queue size for global_level, batch_level and global_multi_level selection"}
    )
    kd_temp: float = field(
        default=1,
        metadata={"help": "teacher/student model temperature for distillation"}
    )
    alpha: Optional[float] = field(
        default=0,
        metadata={"help": "weightage for KD loss, 0 means pure training without KD"}
    )
    use_adaptive_kd_rates: bool = field(
        default=False,
        metadata={"help": "whether to use adaptive distil rate, i.e. different distil rates for different languages"}
    )
    kd_queue_sampling_temp: Optional[float] = field(
        default=None,
        metadata={"help": "temperature value for generating distil rates"}
    )



@register_criterion(
    "label_smoothed_cross_entropy_with_kd", dataclass=KDLabelSmoothedCrossEntropyCriterionConfig
)
class KDLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        kd_rate,
        kd_queue_size,
        kd_temp=1,
        alpha=0.5,
        use_adaptive_kd_rates=False,
        kd_queue_sampling_temp=1.5,
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
        self.kd_strategy = self.task.kd_strategy
        self.kd_temp = kd_temp
        self.kd_rate = kd_rate
        self.alpha = alpha
        self.kd_queue_size = kd_queue_size
        self.num_languages = len(self.task.src_lang_ids)
        self.use_adaptive_kd_rates = use_adaptive_kd_rates
        self.kd_queue_sampling_temp = kd_queue_sampling_temp
        self.kd_lang_wise_count = defaultdict(int)

        if self.kd_strategy == "global_language_wise":
            self.queue = {}
            for id in self.task.src_lang_ids:
                self.queue[id] = torch.cuda.FloatTensor([])
        else:
            self.queue = torch.cuda.FloatTensor([])

    
    def get_lang_kd_rates(self):
        if self.use_adaptive_kd_rates:
            lens = np.array(list(self.kd_lang_wise_count.values()))
            lens_prob = np.power(lens/lens.sum(), 1/self.kd_queue_sampling_temp)
            return lens_prob
        else:
            return [self.kd_rate] * len(self.kd_lang_wise_count)


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
            self.queue = self.queue[tensor_sz: ]
        self.queue = torch.cat((self.queue, tensor))


    def push_to_lang_FIFO_queue(self, id, tensor):
        # this method is applicable only when we have a mulitple global queues
        # here self.queue is dictionary of torch.cuda.FloatTensors
        tensor = tensor.detach()
        tensor_sz = tensor.size(0)
        current_queue_sz = self.queue[id].size(0)
        if tensor_sz + current_queue_sz > self.kd_queue_size:
            self.queue[id] = self.queue[id][tensor_sz: ]
        self.queue[id] = torch.cat((self.queue[id], tensor))


    def forward(self, model, sample, update_num=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        teacher_output = sample.get("teacher_output", None)

        assert teacher_output is not None, "knowledge distillation requires a teacher output!"

        loss, extra = self.compute_loss(
            model, 
            net_output, 
            sample, 
            teacher_output=teacher_output
        )

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'kd_loss': extra['kd_loss'].data if extra.get('kd_loss', None) is not None else 0,
            'nll_loss': extra['nll_loss'].data if extra.get('nll_loss', None) is not None else loss.data,
        }
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


    def compute_loss(self, model, net_output, sample, teacher_output=None):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        pad_mask = target.eq(self.padding_idx).view(-1)
        extra = {}

        # get student logits
        student_logits = net_output[0]
        student_logits = student_logits.view(-1, student_logits.size(-1))
        student_logits_T = student_logits/self.kd_temp

        # get teacher probs and lprobs
        teacher_logits = teacher_output[0]
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        teacher_probs_T = F.softmax(teacher_logits/self.kd_temp, dim=-1)
        teacher_lprobs = F.log_softmax(teacher_logits, dim=-1)

        # compute preliminary loss and nll_loss of student_model
        golden_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, 
            target, 
            self.eps, 
            reduce=False,
            ignore_index=self.padding_idx
        )

        nll_loss = nll_loss.view(-1)
        golden_loss = golden_loss.view(-1)

        kd_loss = F.cross_entropy(
            student_logits_T,
            teacher_probs_T,
            reduction='none'
        ).masked_fill_(pad_mask, 0).view(-1)

        if self.kd_strategy == 'word_and_seq_level':
            extra['kd_loss'] = kd_loss.sum()
            extra['nll_loss'] = nll_loss.sum()
            loss = (1 - self.alpha) * golden_loss.sum() + self.alpha * (self.kd_temp ** 2) * extra['kd_loss']

        elif self.kd_strategy == 'batch_level':
            loss_gate = nll_loss.topk(
                math.ceil(
                    nll_loss.size(0) * self.kd_rate
                ), 
                dim=0, 
                largest=True
            )[0][-1]
            KD_mask = nll_loss >= loss_gate
            extra['kd_loss'] = kd_loss[KD_mask].sum()
            extra['nll_loss'] = nll_loss.sum()
            loss = (1 - self.alpha) * golden_loss.sum() + self.alpha * (self.kd_temp ** 2) * extra['kd_loss']
            
        elif self.kd_strategy == 'global_level':
            self.push_to_FIFO_queue(nll_loss)
            loss_gate = self.queue.topk(
                math.ceil(
                    self.queue.size(0) * self.kd_rate
                ), 
                dim=0, 
                largest=True
            )[0][-1]
            KD_mask = nll_loss >= loss_gate # B * T
            extra['kd_loss'] = kd_loss[KD_mask].sum()
            extra['nll_loss'] = nll_loss.sum()
            loss = (1 - self.alpha) * golden_loss.sum() + self.alpha * (self.kd_temp ** 2) * extra['kd_loss']

        elif self.kd_strategy == "global_language_wise":
            indices, kd_loss_ = defaultdict(list), 0
            nll_loss_langwise, kd_loss_langwise = {}, {}
            inp_tokens = sample["net_input"]["src_tokens"]

            for idx, lang_id in enumerate(self.get_lang_ids(inp_tokens)):
                indices[lang_id].append(idx)
                self.kd_lang_wise_count[lang_id] += 1

            for lang_id, idx in indices.items():
                idx = torch.cuda.LongTensor(idx)
                nll_loss_lang = nll_loss.index_select(0, idx).view(-1)
                kd_loss_lang = kd_loss.index_select(0, idx).view(-1)
                nll_loss_langwise[lang_id] = nll_loss_lang
                kd_loss_langwise[lang_id] = kd_loss_lang
                self.push_to_lang_FIFO_queue(lang_id, nll_loss_lang)
            kd_rates = self.get_lang_kd_rates()
            
            for (lang_id, kd_rate) in zip(indices.keys(), kd_rates):
                loss_gate = self.queue[lang_id].topk(
                    math.ceil(
                        self.queue[lang_id].size(0) * kd_rate
                    ), 
                    dim=0, 
                    largest=True
                )[0][-1]
                KD_mask = nll_loss_langwise[lang_id] >= loss_gate
                kd_loss_ += kd_loss_langwise[lang_id][KD_mask].sum()

            extra['kd_loss'] = kd_loss_
            extra['nll_loss'] = nll_loss.sum()
            loss = (1 - self.alpha) * golden_loss.sum() + self.alpha * (self.kd_temp ** 2) * extra['kd_loss']

        else:
            raise ValueError("unknown strategy or parameter mismatch")
        return loss, extra


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # sum metrics
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        kd_loss = sum(log.get('kd_loss', 0) for log in logging_outputs)
        # log metrics
        metrics.log_scalar(
            'loss', 
            loss / sample_size / math.log(2), 
            sample_size, 
            round=3
        )
        metrics.log_scalar(
            'nll_loss', 
            nll_loss / ntokens / math.log(2), 
            ntokens, 
            round=3)
        metrics.log_scalar(
            'kd_loss', 
            kd_loss / ntokens / math.log(2), 
            ntokens, 
            round=3)
        metrics.log_derived(
            'ppl', 
            lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

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
