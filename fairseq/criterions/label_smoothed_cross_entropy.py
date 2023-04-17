# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np 
import torch.nn.functional as F
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import queue

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.range_eps = 0.01
        self.queue = torch.cuda.FloatTensor([])
        self.teacher_loss_queue =  torch.cuda.FloatTensor([])
        self.real_distil_rate = 0.0
        self.dict_count = None

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on
    
    def push_to_FIFO_queue(self, tensor):
        tensor = tensor.detach().view(-1)
        tensor_size = tensor.size(0)
        current_size = self.queue.size(0)
        self.queue = self.queue.view(-1)
        if tensor_size + current_size < self.task.args.difficult_queue_size:
            self.queue = torch.cat((self.queue, tensor))
        else:
            self.queue = torch.cat((self.queue[tensor_size: ], tensor))
    
    def push_to_teacher_FIFO_queue(self, tensor):
        tensor = tensor.detach().view(-1)
        tensor_size = tensor.size(0)
        current_size = self.teacher_loss_queue.size(0)
        self.teacher_loss_queue = self.teacher_loss_queue.view(-1)
        if tensor_size + current_size < self.task.args.difficult_queue_size:
            self.teacher_loss_queue = torch.cat((self.teacher_loss_queue, tensor))
        else:
            self.teacher_loss_queue = torch.cat((self.teacher_loss_queue[tensor_size: ], tensor))

    def forward(self, model, sample, reduce=True, teacher_model=None, update_num=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        teacher_output = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(**sample['net_input'])

        loss, nll_loss, extra_result = self.compute_loss(model, net_output, sample, reduce=reduce, 
                                            teacher_output=teacher_output, 
                                            distil_strategy=self.task.args.distil_strategy,
                                            update_num=update_num)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data if nll_loss is not None else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'distil_rate': self.real_distil_rate,
            'gpu_nums':1,
            'KD_loss': extra_result['KD_loss'].data if extra_result.get('KD_loss', None) is not None else 0,  
            'nll_loss_distil': extra_result['nll_loss_distil'].data if extra_result.get('nll_loss_distil', None) is not None else 0,  
            #'distil_token_num': extra_result['distil_token_num'].data if extra_result.get('distil_token_num', None) is not None else 0,  
        }
        
        return loss, sample_size, logging_output

    def get_teacher_probs(self, teacher_output):
        teacher_predict = teacher_output[0]
        teacher_predict = teacher_predict.view(-1, teacher_predict.size(-1)) # B*T x vocab
        if self.task.args.teacher_predict_temperature_schedule == 'binary':
            teacher_predict_max = torch.max(teacher_predict, dim=-1)[0].view(-1, 1) # B*T x 1
            teacher_predict_mask = teacher_predict_max > 0.5 # B*T x 1
            temperature = torch.ones_like(teacher_predict_max) / self.task.args.teacher_predict_temperature # B*T x 1 
            temperature = temperature.masked_fill(teacher_predict_mask, self.task.args.teacher_predict_temperature) # B*T x 1
            teacher_predict = teacher_predict * temperature
        elif self.task.args.teacher_predict_temperature_schedule == 'topk':
            distil_lprobs = F.softmax(teacher_predict, dim=-1, dtype=torch.float32) # B * T x vocab
            distil_mask = distil_lprobs > 0.01
            invalid_mask = distil_mask.sum(dim=-1) == 0
            distil_mask[invalid_mask, :] = True
            teacher_predict.masked_fill_(~distil_mask, float("-inf"))
        else:
            teacher_predict = teacher_predict * self.task.args.teacher_predict_temperature
        distil_lprobs = F.softmax(teacher_predict, dim=-1, dtype=torch.float32) # B x T x vocab
        return distil_lprobs

    def compute_loss(self, model, net_output, sample, reduce=True, teacher_output=None, distil_strategy="normal", update_num=None):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = torch.log(probs)
        probs = probs.view(-1, lprobs.size(-1))
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        bsz, seq_len = target.shape
        target = target.view(-1, 1)
        pad_mask = target.eq(self.padding_idx).view(-1)
        loss = None
        nll_loss = None
        extra_result = {}
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss.masked_fill_(pad_mask, 0.)
            KL_loss = KL_loss.sum()
            extra_result['KD_loss'] = KL_loss
            loss = golden_loss + KL_loss 
        elif distil_strategy == 'distil_only':
            # only get supervision signal from KD
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
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
            word_rate = self.task.args.distil_rate
            self.real_distil_rate = word_rate
            loss_gate = nll_loss.topk(math.ceil(words_num * word_rate), dim=0, largest=True)[0][-1]
            KL_mask = nll_loss < loss_gate
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            KL_loss = KL_loss[~KL_mask]
            KD_loss = KL_loss.sum() 
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'global_level':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            teacher_predicts = self.get_teacher_probs(teacher_output)
            nll_loss = nll_loss.view(-1) 
            golden_loss = golden_loss.view(-1)
            words_num = nll_loss.size(0)
            word_rate = self.task.args.distil_rate
            teacher_predicts = teacher_predicts[~pad_mask]
            lprobs = lprobs[~pad_mask]
            target = target[~pad_mask]
            nll_loss = nll_loss[~pad_mask]
            golden_loss = golden_loss[~pad_mask]
            # get kl loss
            KL_loss = F.kl_div(lprobs, teacher_predicts, reduction='none')
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
            loss = golden_loss.sum() + KD_loss
      
        elif distil_strategy == 'word_ce_low':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            distil_lprobs = self.get_teacher_probs(teacher_output)
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            words_num = nll_loss.size(0)
            word_rate = self.task.args.distil_rate
            self.real_distil_rate = word_rate
            loss_gate = nll_loss.topk(math.ceil(words_num * word_rate), dim=0, largest=False)[0][-1]
            KL_mask = nll_loss > loss_gate
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            KL_loss = KL_loss[~KL_mask]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'word_ce_high':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            distil_lprobs = self.get_teacher_probs(teacher_output)
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            words_num = nll_loss.size(0)
            word_rate = self.task.args.distil_rate
            self.real_distil_rate = word_rate
            loss_gate = nll_loss.topk(math.ceil(words_num * word_rate), dim=0, largest=True)[0][-1]
            KL_mask = nll_loss < loss_gate
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            KL_loss = KL_loss[~KL_mask]
            KD_loss = KL_loss.sum() 
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            word_emb = model.decoder.embed_tokens.weight.data.detach()
            words_emb_norm = word_emb.norm(dim=-1)
            words_emb_norm = words_emb_norm.view(-1, 1) # n_vocab x 1
            target_norm = words_emb_norm[target].view(-1)
            target_norm = target_norm[~pad_mask] # 
            norm_gate = target_norm.topk(math.ceil(word_num * self.task.args.distil_rate), dim=0, largest=True)[0][-1]
            need_learn = norm_gate <= target_norm
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            word_emb = model.decoder.embed_tokens.weight.data.detach()
            words_emb_norm = word_emb.norm(dim=-1)
            words_emb_norm = words_emb_norm.view(-1, 1) # n_vocab x 1
            target_norm = words_emb_norm[target].view(-1)
            target_norm = target_norm[~pad_mask] # 
            norm_gate = target_norm.topk(math.ceil(word_num * self.task.args.distil_rate), dim=0, largest=False)[0][-1]
            need_learn = norm_gate >= target_norm
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]

            # get gate
            target_numpy = target.data.cpu().numpy()
            target_count = self.dict_count[target_numpy] 
            target_count_cuda = torch.tensor(target_count).to(target)
            target_count_cuda = target_count_cuda.view(-1)
            target_count_cuda = target_count_cuda[~pad_mask]
            count_gate = target_count_cuda.topk(math.ceil(word_num * self.task.args.distil_rate), dim=0, largest=False)[0][-1]
            
            need_learn = target_count_cuda <= count_gate
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(-1)
            KL_loss = KL_loss[~pad_mask]

            # get gate
            target_numpy = target.data.cpu().numpy()
            target_count = self.dict_count[target_numpy] 
            target_count_cuda = torch.tensor(target_count).to(target)
            target_count_cuda = target_count_cuda.view(-1)
            target_count_cuda = target_count_cuda[~pad_mask]
            count_gate = target_count_cuda.topk(math.ceil(word_num * self.task.args.distil_rate), dim=0, largest=True)[0][-1]
            
            need_learn = target_count_cuda >= count_gate
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(bsz, seq_len)

            # get gate 
            pad_mask = pad_mask.view(bsz, seq_len)
            seq_real_len = (~pad_mask).sum(dim=-1).view(-1)
            sentence_gate = seq_real_len.topk(math.ceil(bsz * self.task.args.distil_rate), dim=0, largest=True)[0][-1]
            need_ignore = seq_real_len < sentence_gate
            need_ignore = need_ignore.view(-1, 1)
            KL_loss.masked_fill_(need_ignore, 0.0)

            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(bsz, seq_len)

            # get gate 
            pad_mask = pad_mask.view(bsz, seq_len)
            seq_real_len = (~pad_mask).sum(dim=-1).view(-1)
            sentence_gate = seq_real_len.topk(math.ceil(bsz * self.task.args.distil_rate), dim=0, largest=False)[0][-1]
            need_ignore = seq_real_len > sentence_gate
            need_ignore = need_ignore.view(-1, 1)
            KL_loss.masked_fill_(need_ignore, 0.0)

            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(bsz, seq_len)
            
            # get gate 
            pad_mask = pad_mask.view(bsz, seq_len)
            seq_real_len = (~pad_mask).sum(dim=-1).view(-1)
            golden_loss = golden_loss.view(bsz, seq_len)
            golden_loss_mean = golden_loss.sum(dim=-1).view(-1) / seq_real_len.float()
            sentence_gate = golden_loss_mean.topk(math.ceil(bsz * self.task.args.distil_rate), dim=0, largest=True)[0][-1]
            need_ignore = golden_loss_mean > sentence_gate

            need_ignore = need_ignore.view(-1, 1)
            KL_loss.masked_fill_(need_ignore, 0.0)
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
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
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss = KL_loss.view(bsz, seq_len)
            
            # get gate 
            pad_mask = pad_mask.view(bsz, seq_len)
            seq_real_len = (~pad_mask).sum(dim=-1).view(-1)
            golden_loss = golden_loss.view(bsz, seq_len)
            golden_loss_mean = golden_loss.sum(dim=-1).view(-1) / seq_real_len.float()
            sentence_gate = golden_loss_mean.topk(math.ceil(bsz * self.task.args.distil_rate), dim=0, largest=False)[0][-1]
            need_ignore = golden_loss_mean < sentence_gate

            need_ignore = need_ignore.view(-1, 1)
            KL_loss.masked_fill_(need_ignore, 0.0)
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
            nll_loss = nll_loss.sum()
        elif distil_strategy == 'teacher_entropy_low':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1).view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            entropy = -1.0 * (distil_lprobs * torch.log(distil_lprobs)).sum(dim=-1)
            entropy = entropy.view(-1)
            entropy = entropy[~pad_mask]
            entropy_gate = entropy.topk(math.ceil(word_num * self.task.args.distil_rate), dim=0, largest=False)[0][-1]
            need_learn = entropy < entropy_gate
            need_learn = need_learn.view(-1)
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
            nll_loss = nll_loss.sum()
        elif distil_strategy == 'teacher_entropy_high':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1).view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            entropy = -1.0 * (distil_lprobs * torch.log(distil_lprobs)).sum(dim=-1)
            entropy = entropy.view(-1)
            entropy = entropy[~pad_mask]
            entropy_gate = entropy.topk(math.ceil(bsz * self.task.args.distil_rate), dim=0, largest=True)[0][-1]
            need_learn = entropy > entropy_gate
            need_learn = need_learn.view(-1)
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            loss = golden_loss.sum() + KD_loss
            nll_loss = nll_loss.sum()

        elif distil_strategy == 'teacher_golden_high':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1).view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            teacher_golden_predict = distil_lprobs.gather(dim=-1, index=target)
            teacher_golden_predict = teacher_golden_predict.view(-1)
            teacher_golden_predict = teacher_golden_predict[~pad_mask]
            golden_gate = teacher_golden_predict.topk(math.ceil(word_num * self.task.args.distil_rate), dim=0, largest=True)[0][-1]
            need_learn = teacher_golden_predict > golden_gate
            need_learn = need_learn.view(-1)
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
            nll_loss = nll_loss.sum()
        
        elif distil_strategy == 'teacher_golden_low':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            nll_loss = nll_loss.view(-1)
            nll_loss = nll_loss[~pad_mask]
            word_num = nll_loss.size(0)
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1).view(-1)
            KL_loss = KL_loss[~pad_mask]
            # get gate
            teacher_golden_predict = distil_lprobs.gather(dim=-1, index=target)
            teacher_golden_predict = teacher_golden_predict.view(-1)
            teacher_golden_predict = teacher_golden_predict[~pad_mask]
            golden_gate = teacher_golden_predict.topk(math.ceil(word_num * self.task.args.distil_rate), dim=0, largest=False)[0][-1]
            need_learn = teacher_golden_predict < golden_gate
            need_learn = need_learn.view(-1)
            KL_loss = KL_loss[need_learn]
            KD_loss = KL_loss.sum()
            extra_result['KD_loss'] = KD_loss
            loss = golden_loss.sum() + KD_loss
            nll_loss = nll_loss.sum()

        return loss, nll_loss, extra_result

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        #kd_loss_sum = sum(log.get('KD_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nll_loss_distil = sum(log.get('nll_loss_distil', 0) for log in logging_outputs)
        distil_token_num = sum(log.get('distil_token_num', 0) for log in logging_outputs)
        GPU_nums = sum(log.get('gpu_nums', 0) for log in logging_outputs)
        real_distil_rate = sum(log.get('distil_rate', 0) for log in logging_outputs) / GPU_nums
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        #metrics.log_scalar('kd_loss_sum', kd_loss_sum / distil_token_num, round=4)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('distil_rate', real_distil_rate, round=4)
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
