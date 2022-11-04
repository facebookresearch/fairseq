# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from typing import Optional
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class KDLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    kd_rate: Optional[float] = field(
        default=None,
        metadata={"help": "the hyperparameter `tau` to control the number of words to get distillation knowledge"}
    )
    kd_queue_size: Optional[int] = field(
        default=20000, 
        metadata={"help": "queue size for global_level, batch_level and global_multi_level selection"}
    )
    student_temp: float = field(
        default=1,
        metadata={"help": "student model temperature for distillation"}
    )
    teacher_temp: float = field(
        default=1,
        metadata={"help": "teacher model emperature for distillation"}
    )
    alpha: Optional[float] = field(
        default=1,
        metadata={"help": "weightage for KD loss, 0 means pure training without KD"}
    )
    beta: Optional[float] = field(
        default=0,
        metadata={"help": "weightage for cosine similarity loss"}
    )
    use_adaptive_kd_rates: bool = field(
        default=False,
        metadata={"help": "whether to use adaptive distil rate, i.e. different distil rates for different languages"}
    )
    kd_queue_sampling_temp: Optional[float] = field(
        default=None,
        metadata={"help": "temperature value for generating distil rates"}
    )
    use_encoder_cosine_similarity_loss: bool = field(
        default=False,
        metadata={"help": "add encoder cosine similarity loss while performing kd"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


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
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_with_kd", dataclass=KDLabelSmoothedCrossEntropyCriterionConfig
)
class KDLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        kd_rate,
        kd_queue_size,
        student_temp,
        teacher_temp,
        alpha,
        beta,
        use_adaptive_kd_rates,
        kd_queue_sampling_temp,
        use_encoder_cosine_similarity_loss,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        
        # new parameters
        self.kd_strategy = self.task.kd_strategy
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.kd_rate = kd_rate
        self.alpha = alpha
        self.beta = beta
        self.kd_queue_size = kd_queue_size
        self.num_languages = len(self.task.src_lang_ids)
        self.use_adaptive_kd_rates = use_adaptive_kd_rates
        self.kd_queue_sampling_temp = kd_queue_sampling_temp
        self.use_encoder_cosine_similarity_loss = use_encoder_cosine_similarity_loss

        if self.kd_strategy == "global_multi_level":
            self.queue = {}
            for id in self.task.src_lang_ids:
                self.queue[id] = torch.cuda.FloatTensor([])
        else:
            self.queue = torch.cuda.FloatTensor([])

    
    def get_lang_kd_rates(self, indices, T=1):
        if self.use_adaptive_kd_rates:
            lens = torch.cuda.FloatTensor([len(v) for v in indices.values()])
            lens_prob = F.softmax(1/(lens*T), dim=-1).tolist()
            return lens_prob
        else:
            return [self.kd_rate] * len(indices)


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


    def forward(self, model, sample, reduce=True):
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
            teacher_output=teacher_output)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'kd_loss': extra['kd_loss'].data if extra.get('kd_loss', None) is not None else 0,
            'cos_sim_loss': extra['cos_sim_loss'].data if extra.get('cos_sim_loss', None) is not None else 0,
            'nll_loss_student': extra['nll_loss_student'].data if extra.get('nll_loss_student', None) is not None else loss.data,
            'nll_loss_teacher': extra['nll_loss_teacher'].data if extra.get('nll_loss_teacher', None) is not None else 0
        }
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    def encoder_cosine_similarity_loss(self, teacher_encoder_output, student_encoder_output):
        enc_pad_mask = teacher_encoder_output["encoder_padding_mask"][0].view(-1)

        h_s = student_encoder_output["encoder_out"][0]
        h_t = teacher_encoder_output["encoder_out"][0]

        assert h_s.size(-1) == h_t.size(-1), f"student ({h_s.size(-1)}) and teacher ({h_t.size(-1)}) model are of different dimensions"

        h_t = h_t.contiguous().view(-1, h_t.size(-1))
        h_s = h_s.contiguous().view(-1, h_s.size(-1))

        return F.cosine_embedding_loss(
            h_s, h_t,
            torch.ones(
                h_s.size(0), 
                device="cuda"
            ),
            reduction='none'
        ).masked_fill_(enc_pad_mask, 0).sum()


    def compute_loss(self, model, net_output, sample, teacher_output=None):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        pad_mask = target.eq(self.padding_idx).view(-1)
        extra = dict()

        # get student logits
        student_logits = net_output[0]
        student_logits = student_logits.view(-1, student_logits.size(-1))
        student_logits_T = student_logits/self.student_temp

        # get teacher probs and lprobs
        teacher_logits = teacher_output[0]
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        teacher_probs_T = F.softmax(teacher_logits/self.teacher_temp, dim=-1)
        teacher_lprobs = F.log_softmax(teacher_logits, dim=-1)

        # compute preliminary loss and nll_loss of student_model
        golden_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, 
            target, 
            self.eps, 
            reduce=False,
            ignore_index=self.padding_idx
        )

        if teacher_lprobs is not None:
            # compute preliminary lprobs, loss, nll_loss of teacher_model
            teacher_lprobs = teacher_lprobs.view(-1, teacher_lprobs.size(-1))
            _, nll_loss_teacher = label_smoothed_nll_loss(
                teacher_lprobs, 
                target, 
                self.eps, 
                reduce=False,
                ignore_index=self.padding_idx
            )

        nll_loss = nll_loss.view(-1)
        nll_loss_teacher = nll_loss_teacher.view(-1)
        golden_loss = golden_loss.view(-1)

        if self.use_encoder_cosine_similarity_loss:
            # get the student and teacher encoder representations
            teacher_encoder_output = sample.get("teacher_encoder_output", None)
            student_encoder_output = model.get_encoder_output()
            extra['cos_sim_loss'] = self.encoder_cosine_similarity_loss(
                teacher_encoder_output, student_encoder_output
            )

        kd_loss = F.cross_entropy(
            student_logits_T,
            teacher_probs_T,
            reduction='none'
        ).masked_fill_(pad_mask, 0).view(-1)

        if self.kd_strategy == 'word_and_seq_level':
            extra['kd_loss'] = kd_loss.sum()
            extra['nll_loss_student'] = nll_loss.sum()
            extra['nll_loss_teacher'] = nll_loss_teacher.sum()
            loss = golden_loss.sum() + \
                   self.alpha * self.student_temp * self.teacher_temp * extra['kd_loss'] + \
                   self.beta * extra.get('cos_sim_loss', 0)

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
            extra['nll_loss_student'] = nll_loss.sum()
            extra['nll_loss_teacher'] = nll_loss_teacher.sum()
            loss = golden_loss.sum() + \
                   self.alpha * self.student_temp * self.teacher_temp * extra['kd_loss'] + \
                   self.beta * extra.get('cos_sim_loss', 0)
            
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
            extra['nll_loss_student'] = nll_loss.sum()
            extra['nll_loss_teacher'] = nll_loss_teacher.sum()
            loss = golden_loss.sum() + \
                   self.alpha * self.student_temp * self.teacher_temp * extra['kd_loss'] + \
                   self.beta * extra.get('cos_sim_loss', 0)

        elif self.kd_strategy == "global_multi_level":
            indices, total_kd_loss = dict(), 0
            inp_tokens = sample["net_input"]["src_tokens"]
            for idx, val in enumerate(self.get_lang_ids(inp_tokens)):
                indices.setdefault(val, []).append(idx)
            nll_loss = nll_loss.view(inp_tokens.size(0), -1)
            for key, val in indices.items():
                nll_loss_lang = nll_loss.index_select(0, torch.cuda.LongTensor(val)).view(-1)
                self.push_to_lang_FIFO_queue(key, nll_loss_lang)
            kd_rates = self.get_lang_kd_rates(indices, self.kd_queue_sampling_temp)
            
            for idx, kd_rate in zip(indices.keys(), kd_rates):
                loss_gate = self.queue[idx].topk(
                    math.ceil(
                        self.queue[idx].size(0) * kd_rate
                    ), 
                    dim=0, 
                    largest=True
                )[0][-1]
                KD_mask = nll_loss_lang >= loss_gate
                KD_indices = KD_mask.nonzero().view(-1)
                total_kd_loss += kd_loss.gather(0, KD_indices).sum()

            extra['kd_loss'] = total_kd_loss
            extra['nll_loss_student'] = nll_loss.sum()
            extra['nll_loss_teacher'] = nll_loss_teacher.sum()
            loss = golden_loss.sum() + \
                   self.alpha * self.student_temp * self.teacher_temp * extra['kd_loss'] + \
                   self.beta * extra.get('cos_sim_loss', 0)

        else:
            raise ValueError("unknown strategy or parameter mismatch")
        return loss, extra


    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # sum metrics
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_student = sum(log.get('nll_loss_student', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nll_loss_teacher = sum(log.get('nll_loss_teacher', 0) for log in logging_outputs)
        kd_loss = sum(log.get('kd_loss', 0) for log in logging_outputs)
        cos_sim_loss = sum(log.get('cos_sim_loss', 0) for log in logging_outputs)
        # log metrics
        metrics.log_scalar(
            'loss', 
            loss / sample_size / math.log(2), 
            sample_size, 
            round=3
        )
        metrics.log_scalar(
            'nll_loss', 
            nll_loss_student / ntokens / math.log(2), 
            ntokens, 
            round=3)
        metrics.log_scalar(
            'nll_loss_teacher', 
            nll_loss_teacher / ntokens / math.log(2), 
            ntokens, 
            round=3)
        metrics.log_scalar(
            'kd_loss', 
            kd_loss / ntokens / math.log(2), 
            ntokens, 
            round=3)
        metrics.log_scalar(
            'cos_sim_loss', 
            cos_sim_loss / ntokens,
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


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
