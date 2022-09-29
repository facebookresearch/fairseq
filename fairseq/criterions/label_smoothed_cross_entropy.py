# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
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
    aplha_kd: float = field(
        default=0,
        metadata={"help": "KD loss weightage, 0 means pure training without KD"}
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
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        alpha_kd,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        # new parameters
        self.range_eps = 0.01
        self.queue = torch.cuda.FloatTensor([])
        self.teacher_loss_queue =  torch.cuda.FloatTensor([])
        self.real_distil_rate = 0.0
        self.dict_count = None
        self.alpha_kd = alpha_kd


    def push_to_FIFO_queue(self, tensor):
        tensor = tensor.detach().view(-1)
        tensor_size = tensor.size(0)
        current_size = self.queue.size(0)
        self.queue = self.queue.view(-1)
        if tensor_size + current_size < self.task.difficult_queue_size:
            self.queue = torch.cat((self.queue, tensor))
        else:
            self.queue = torch.cat((self.queue[tensor_size: ], tensor))
    
    
    def push_to_teacher_FIFO_queue(self, tensor):
        tensor = tensor.detach().view(-1)
        tensor_size = tensor.size(0)
        current_size = self.teacher_loss_queue.size(0)
        self.teacher_loss_queue = self.teacher_loss_queue.view(-1)
        if tensor_size + current_size < self.task.difficult_queue_size:
            self.teacher_loss_queue = torch.cat((self.teacher_loss_queue, tensor))
        else:
            self.teacher_loss_queue = torch.cat((self.teacher_loss_queue[tensor_size: ], tensor))


    def get_teacher_probs(self, teacher_output):
        teacher_predict = teacher_output[0]
        teacher_predict = teacher_predict.view(-1, teacher_predict.size(-1)) # B*T x vocab
        if self.task.temperature_schedule == 'binary':
            teacher_predict_max = torch.max(teacher_predict, dim=-1)[0].view(-1, 1) # B*T x 1
            teacher_predict_mask = teacher_predict_max > 0.5 # B*T x 1
            temperature = torch.ones_like(teacher_predict_max) / self.task.teacher_predict_temperature # B*T x 1 
            temperature = temperature.masked_fill(teacher_predict_mask, self.task.teacher_predict_temperature) # B*T x 1
            teacher_predict /= self.task.temperature
        elif self.task.temperature_schedule == 'topk':
            distil_probs = F.softmax(teacher_predict, dim=-1) # B * T x vocab
            distil_mask = distil_probs > 0.01
            invalid_mask = (distil_mask.sum(dim=-1) == 0)
            distil_mask[invalid_mask, :] = True
            teacher_predict.masked_fill_(~distil_mask, float("-inf"))
        else:
            teacher_predict /= self.task.temperature
        distil_lprobs = F.log_softmax(teacher_predict, dim=-1) # B x T x vocab
        return distil_lprobs


    def forward(self, model, sample, reduce=True, teacher_model=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        teacher_output = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(**sample['net_input'])

        loss, nll_loss, extra_result = self.compute_loss(
            model, 
            net_output, 
            sample, 
            reduce=reduce, 
            teacher_output=teacher_output, 
            distil_strategy=self.task.distillation_strategy)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data if nll_loss is not None else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'distil_rate': self.real_distil_rate,
            'KD_loss': extra_result['KD_loss'].data if extra_result.get('KD_loss', None) is not None else 0,
            'num_distil_token': extra_result['num_distil_token'].data if extra_result.get('num_distil_token', None) is not None else 0
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


    def compute_loss(self, model, net_output, sample, reduce=True, teacher_output=None, distil_strategy="normal"):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        bsz, seq_len = target.size()
        target = target.view(-1, 1)
        pad_mask = target.eq(self.padding_idx).view(-1)
        loss, nll_loss, extra_result = None, None, {}

        if distil_strategy == 'normal' or teacher_output is None:
            # not use distillation
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )

        elif distil_strategy == 'distil_all':
            # distill all word with no selection
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss.masked_fill_(pad_mask, 0.)
            KL_loss = KL_loss.sum()
            extra_result['KD_loss'] = KL_loss
            loss = golden_loss + (self.alpha_kd * KL_loss)

        elif distil_strategy == 'distil_only':
            # only get supervision signal from KD
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss.masked_fill_(pad_mask, 0.)
            loss = KL_loss.sum()
            nll_loss = None

        elif distil_strategy == 'batch_level':
            # batch level selection, Word CE
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            distil_lprobs = self.get_teacher_probs(teacher_output)
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            words_num = nll_loss.size(0)
            word_rate = self.task.distillation_rate
            self.real_distil_rate = word_rate
            loss_gate = nll_loss.topk(math.ceil(words_num * word_rate), dim=0, largest=True)[0][-1]
            KL_mask = nll_loss < loss_gate
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            KL_loss = KL_loss[~KL_mask]
            KD_loss = KL_loss.sum() 
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'global_level':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            distil_lprobs = self.get_teacher_probs(teacher_output)
            nll_loss = nll_loss.view(-1) 
            golden_loss = golden_loss.view(-1)
            words_num = nll_loss.size(0)
            word_rate = self.task.distillation_rate
            distil_lprobs = distil_lprobs[~pad_mask]
            lprobs = lprobs[~pad_mask]
            target = target[~pad_mask]
            nll_loss = nll_loss[~pad_mask]
            golden_loss = golden_loss[~pad_mask]
            # get kl loss
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1).view(-1) # B * T 
            # from the queue get the gate
            self.push_to_FIFO_queue(nll_loss)
            loss_gate = self.queue.topk(math.ceil(self.queue.size(0) * word_rate), dim=0, largest=True)[0][-1]
            KL_mask = nll_loss < loss_gate # B * T
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~KL_mask]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            nll_loss = nll_loss.sum()
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
      
        elif distil_strategy == 'word_ce_low':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            distil_lprobs = self.get_teacher_probs(teacher_output)
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            words_num = nll_loss.size(0)
            word_rate = self.task.distillation_rate
            self.real_distil_rate = word_rate
            loss_gate = nll_loss.topk(math.ceil(words_num * word_rate), dim=0, largest=False)[0][-1]
            KL_mask = nll_loss > loss_gate
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            KL_loss = KL_loss[~KL_mask]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'word_ce_high':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            distil_lprobs = self.get_teacher_probs(teacher_output)
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            words_num = nll_loss.size(0)
            word_rate = self.task.distillation_rate
            self.real_distil_rate = word_rate
            loss_gate = nll_loss.topk(math.ceil(words_num * word_rate), dim=0, largest=True)[0][-1]
            KL_mask = nll_loss < loss_gate
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            KL_loss = KL_loss[~KL_mask]
            KD_loss = KL_loss.sum() 
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'word_norm_high':
            # get loss
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            word_emb = model.decoder.embed_tokens.weight.data.detach()
            words_emb_norm = word_emb.norm(dim=-1)
            words_emb_norm = words_emb_norm.view(-1, 1) # n_vocab x 1
            target_norm = words_emb_norm[target].view(-1)
            target_norm = target_norm[~pad_mask] # 
            norm_gate = target_norm.topk(math.ceil(word_num * self.task.distillation_rate), dim=0, largest=True)[0][-1]
            need_learn = norm_gate <= target_norm
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()

        elif distil_strategy == 'word_norm_low':
            # get loss
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            word_emb = model.decoder.embed_tokens.weight.data.detach()
            words_emb_norm = word_emb.norm(dim=-1)
            words_emb_norm = words_emb_norm.view(-1, 1) # n_vocab x 1
            target_norm = words_emb_norm[target].view(-1)
            target_norm = target_norm[~pad_mask] # 
            norm_gate = target_norm.topk(math.ceil(word_num * self.task.distillation_rate), dim=0, largest=False)[0][-1]
            need_learn = norm_gate >= target_norm
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()

        elif distil_strategy == 'word_frequency_low':
            if self.dict_count is None:
                self.dict_count = np.array(self.task.tgt_dict.count)
            # get loss
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]

            # get gate
            target_numpy = target.data.cpu().numpy()
            target_count = self.dict_count[target_numpy] 
            target_count_cuda = torch.tensor(target_count).to(target)
            target_count_cuda = target_count_cuda.view(-1)
            target_count_cuda = target_count_cuda[~pad_mask]
            count_gate = target_count_cuda.topk(math.ceil(word_num * self.task.distillation_rate), dim=0, largest=False)[0][-1]
            
            need_learn = target_count_cuda <= count_gate
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'word_frequency_high':
            if self.dict_count is None:
                self.dict_count = np.array(self.task.tgt_dict.count)
            # get loss
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]

            # get gate
            target_numpy = target.data.cpu().numpy()
            target_count = self.dict_count[target_numpy] 
            target_count_cuda = torch.tensor(target_count).to(target)
            target_count_cuda = target_count_cuda.view(-1)
            target_count_cuda = target_count_cuda[~pad_mask]
            count_gate = target_count_cuda.topk(math.ceil(word_num * self.task.distillation_rate), dim=0, largest=True)[0][-1]
            
            need_learn = target_count_cuda >= count_gate
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'sentence_length_high':
            # get loss
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(bsz, seq_len)

            # get gate 
            pad_mask = pad_mask.view(bsz, seq_len)
            seq_real_len = (~pad_mask).sum(dim=-1).view(-1)
            sentence_gate = seq_real_len.topk(math.ceil(bsz * self.task.distillation_rate), dim=0, largest=True)[0][-1]
            need_ignore = seq_real_len < sentence_gate
            need_ignore = need_ignore.view(-1, 1)
            KL_loss.masked_fill_(need_ignore, 0.0)

            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'sentence_length_low':
            # get loss
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(bsz, seq_len)

            # get gate 
            pad_mask = pad_mask.view(bsz, seq_len)
            seq_real_len = (~pad_mask).sum(dim=-1).view(-1)
            sentence_gate = seq_real_len.topk(math.ceil(bsz * self.task.distillation_rate), dim=0, largest=False)[0][-1]
            need_ignore = seq_real_len > sentence_gate
            need_ignore = need_ignore.view(-1, 1)
            KL_loss.masked_fill_(need_ignore, 0.0)

            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'sentence_loss_mean_low':
            # get loss
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(bsz, seq_len)
            
            # get gate 
            pad_mask = pad_mask.view(bsz, seq_len)
            seq_real_len = (~pad_mask).sum(dim=-1).view(-1)
            golden_loss = golden_loss.view(bsz, seq_len)
            golden_loss_mean = golden_loss.sum(dim=-1).view(-1) / seq_real_len.float()
            sentence_gate = golden_loss_mean.topk(math.ceil(bsz * self.task.distillation_rate), dim=0, largest=True)[0][-1]
            need_ignore = golden_loss_mean > sentence_gate

            need_ignore = need_ignore.view(-1, 1)
            KL_loss.masked_fill_(need_ignore, 0.0)
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'sentence_loss_mean_high':
            # get loss
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(bsz, seq_len)
            
            # get gate 
            pad_mask = pad_mask.view(bsz, seq_len)
            seq_real_len = (~pad_mask).sum(dim=-1).view(-1)
            golden_loss = golden_loss.view(bsz, seq_len)
            golden_loss_mean = golden_loss.sum(dim=-1).view(-1) / seq_real_len.float()
            sentence_gate = golden_loss_mean.topk(math.ceil(bsz * self.task.distillation_rate), dim=0, largest=False)[0][-1]
            need_ignore = golden_loss_mean < sentence_gate

            need_ignore = need_ignore.view(-1, 1)
            KL_loss.masked_fill_(need_ignore, 0.0)
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        elif distil_strategy == 'teacher_entropy_low':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1).view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            entropy = -1.0 * (distil_lprobs * torch.log(distil_lprobs)).sum(dim=-1)
            entropy = entropy.view(-1)
            entropy = entropy[~pad_mask]
            entropy_gate = entropy.topk(math.ceil(word_num * self.task.distillation_rate), dim=0, largest=False)[0][-1]
            need_learn = entropy < entropy_gate
            need_learn = need_learn.view(-1)
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        elif distil_strategy == 'teacher_entropy_high':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1).view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            entropy = -1.0 * (distil_lprobs * torch.log(distil_lprobs)).sum(dim=-1)
            entropy = entropy.view(-1)
            entropy = entropy[~pad_mask]
            entropy_gate = entropy.topk(math.ceil(bsz * self.task.distillation_rate), dim=0, largest=True)[0][-1]
            need_learn = entropy > entropy_gate
            need_learn = need_learn.view(-1)
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()

        elif distil_strategy == 'teacher_golden_high':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1).view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            teacher_golden_predict = distil_lprobs.gather(dim=-1, index=target)
            teacher_golden_predict = teacher_golden_predict.view(-1)
            teacher_golden_predict = teacher_golden_predict[~pad_mask]
            golden_gate = teacher_golden_predict.topk(math.ceil(word_num * self.task.distillation_rate), dim=0, largest=True)[0][-1]
            need_learn = teacher_golden_predict > golden_gate
            need_learn = need_learn.view(-1)
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'teacher_golden_low':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none', log_target=True)
            KL_loss = KL_loss.sum(dim=-1).view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            teacher_golden_predict = distil_lprobs.gather(dim=-1, index=target)
            teacher_golden_predict = teacher_golden_predict.view(-1)
            teacher_golden_predict = teacher_golden_predict[~pad_mask]
            golden_gate = teacher_golden_predict.topk(math.ceil(word_num * self.task.distillation_rate), dim=0, largest=False)[0][-1]
            need_learn = teacher_golden_predict < golden_gate
            need_learn = need_learn.view(-1)
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + (self.alpha_kd * KD_loss)
            nll_loss = nll_loss.sum()

        return loss, nll_loss, extra_result


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
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get('KD_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        # log metrics
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('kd_loss', kd_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

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
