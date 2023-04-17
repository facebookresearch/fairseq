# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.logging import metrics


@register_criterion("guided_label_smoothed_cross_entropy_with_accuracy")
class GuidedCrossEntAccCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        guide_alpha,
        text_input_cost_ratio,
        label_smoothing,
        disable_text_guide_update_num=0,
        attentive_cost_regularization=0,
    ):
        """
        guide_alpha:            alpha to inteplate nll and kd loss
        text_input_cost_ratio:  loss ratio for text only input data
        label_smoothing:        label smoothing ratio
        disable_text_guide_update_num:  only use nll loss for the first N updates
        attentive_cost_regularization:  ratio fo attentive cost
        """
        super().__init__(task)
        self.alpha = guide_alpha
        self.attn_beta = attentive_cost_regularization
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.text_input_cost_ratio = text_input_cost_ratio
        self.disable_update_num = disable_text_guide_update_num
        assert self.alpha >= 0 and self.alpha <= 1.0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: off
        parser.add_argument('--guide-alpha', default=0., type=float, metavar='D',
                            help='alpha to merge kd cost from text to speech input with ce loss')
        # fmt: off
        parser.add_argument('--disable-text-guide-update-num', default=0, type=int, metavar='D',
                            help='disable guided target from text for the first N updates.')
        parser.add_argument("--attentive-cost-regularization", default=0.0, type=float, metavar='D',
                            help="use encoder attentive loss regularization with cost ratio D")
        parser.add_argument("--attentive-cost-without-normalize", action='store_true',
                            help="Don't do normalization during attentive cost computation")

    def forward(self, model, sample, reduce=True):
        reduction = 'sum' if reduce else 'none'
        net_input = sample["net_input"]
        net_output = model(**net_input)
        attn_cost = None
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        is_dual_input = True if net_input['src_tokens'] is not None and net_input.get('src_txt_tokens') is not None else False
        target = model.get_targets(sample, net_output)
        src_token_num = 0
        if is_dual_input:
            # lprobs_spch from speech encoder and lprobs_text from text encoder
            lprobs_spch, lprobs_text = torch.chunk(lprobs, 2)
            lprobs_spch.batch_first = lprobs.batch_first
            lprobs_text.batch_first = lprobs.batch_first

            speech_loss, speech_nll_loss, speech_correct, speech_total = \
                self.guide_loss_and_acc(model, lprobs_spch, lprobs_text, target, reduce=(reduction == 'sum'))
            text_loss, text_nll_loss, text_correct, text_total = self.compute_loss_and_acc(model, lprobs_text, target, reduction=reduction)
            loss = (speech_loss + text_loss)
            nll_loss = (speech_nll_loss + text_nll_loss)
            correct = speech_correct + text_correct
            total = speech_total + text_total

            attn_cost = net_output[1].get('attn_cost')
            if attn_cost is not None:
                # attn_cost is batch_first and padding tokens have been masked already
                src_token_num = attn_cost.ne(0).sum()
                attn_cost = attn_cost.sum()
                loss = loss + attn_cost * self.attn_beta
            else:
                attn_cost = 0
        else:
            loss, nll_loss, correct, total = self.compute_loss_and_acc(model, lprobs, target, reduction=reduction)
            if sample["net_input"]['src_tokens'] is None:   # text input only
                loss = loss * self.text_input_cost_ratio
            speech_loss = None
            speech_nll_loss = None

        sample_size, logging_output = self.get_logging_output(
            sample, loss, nll_loss, correct, total, src_token_num, speech_loss, speech_nll_loss, attn_cost, is_dual_input
        )
        return loss, sample_size, logging_output

    def compute_loss_and_acc(self, model, lprobs, target, reduction='sum'):
        if not lprobs.batch_first:
            lprobs = lprobs.transpose(0, 1)
        lprobs = lprobs.view(-1, lprobs.size(-1))  # -> (B x T) x C
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=(reduction == 'sum'),
        )

        mask = target.ne(self.padding_idx)
        correct = torch.sum(lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return loss, nll_loss, correct, total

    def guide_loss_and_acc(self, model, lprobs, lprobs_teacher, target, reduce=True):
        """ lprobs_teacher is used as guide for lprobs """
        if self.alpha == 0.0 or model.num_updates < self.disable_update_num:
            return self.compute_loss_and_acc(model, lprobs, target, reduction=('sum' if reduce else 'none'))
        if not lprobs.batch_first:
            lprobs = lprobs.transpose(0, 1)
            lprobs_teacher = lprobs_teacher.transpose(0, 1)

        lprobs = lprobs.view(-1, lprobs.size(-1)).float()  # -> (B x T) x C
        lprobs_teacher = lprobs_teacher.view(-1, lprobs_teacher.size(-1)).float()  # -> (B x T) x C
        target = target.view(-1)
        loss = F.nll_loss(lprobs, target, ignore_index=self.padding_idx, reduction='sum' if reduce else 'none')
        nll_loss = loss
        probs_teacher = lprobs_teacher.exp().masked_fill_(target.unsqueeze(-1).eq(self.padding_idx), 0)
        probs_teacher = probs_teacher.detach()
        guide_loss = -(probs_teacher*lprobs).sum() if reduce else -(probs_teacher*lprobs).sum(-1, keepdim=True)
        loss = self.alpha*guide_loss + (1.0 - self.alpha)*loss

        mask = target.ne(self.padding_idx)
        correct = torch.sum(lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return loss, nll_loss, correct, total

    def get_logging_output(
        self,
        sample,
        loss,
        nll_loss,
        correct,
        total,
        src_token_num=0,
        speech_loss=None,
        speech_nll_loss=None,
        attn_cost=None,
        is_dual_input=False,
    ):

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        mul_size = 2 if is_dual_input else 1

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "nll_loss": utils.item(nll_loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"]*mul_size,
            "nsentences": sample["target"].size(0)*mul_size,
            "sample_size": sample_size*mul_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
            "src_token_num": utils.item(src_token_num.data) if src_token_num > 0 else 0,
            "nframes": torch.sum(sample["net_input"]["src_lengths"]).item(),
        }

        if speech_loss is not None:
            logging_output["speech_loss"] = utils.item(speech_loss.data)
            logging_output["speech_nll_loss"] = utils.item(speech_nll_loss.data)
            logging_output["sample_size_speech_cost"] = sample_size
            logging_output["speech_attn_loss"] = attn_cost

        return sample_size*mul_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        src_token_sum = sum(log.get("src_token_num", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)
        speech_loss_sum = sum(log.get("speech_loss", 0) for log in logging_outputs)
        speech_nll_loss_sum = sum(log.get("speech_nll_loss", 0) for log in logging_outputs)
        speech_attn_loss_sum = sum(log.get("speech_attn_loss", 0) for log in logging_outputs)
        sample_size_speech = sum(log.get("sample_size_speech_cost", 0) for log in logging_outputs)

        agg_output = {
            "loss": loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.0,
            "nll_loss": nll_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, and loss
            # is per-sentence loss; else sample_size is ntokens, and the loss
            # becomes per-output token loss
            "speech_loss": speech_loss_sum / sample_size_speech / math.log(2) if sample_size_speech > 0 else 0.0,
            "speech_nll_loss": speech_nll_loss_sum / sample_size_speech / math.log(2) if sample_size_speech > 0 else 0.0,
            "speech_attn_loss": speech_attn_loss_sum / src_token_sum / math.log(2) if src_token_sum > 0 else 0.0,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "nframes": nframes,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            "src_token_num": src_token_sum,
            # total is the number of validate tokens
        }
        return agg_output

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {'nsentences', 'ntokens', 'sample_size'}:
                continue
            metrics.log_scalar(k, v, round=3)
