# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq.models.nat import (
    _apply_del_words,
    _apply_ins_masks,
    _apply_ins_words,
    _fill,
    _skip,
    _skip_encoder_out,
)


class _EnsembleModelEncoder(object):
    def __init__(self, models):
        self.models = models

    def reorder_encoder_out(self, encoder_outs, new_order):
        encoder_outs = [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]
        return encoder_outs


class BasicEnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.bos = self.models[0].decoder.dictionary.bos()
        self.eos = self.models[0].decoder.dictionary.eos()
        self.pad = self.models[0].decoder.dictionary.pad()
        self.unk = self.models[0].decoder.dictionary.unk()
        self.encoder = _EnsembleModelEncoder(self.models)

    def has_encoder(self):
        return hasattr(self.models[0], "encoder")

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.forward_encoder(encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, *inputs):
        raise NotImplementedError

    def initialize_output_tokens(self, *inputs):
        raise NotImplementedError


class EnsembleLevT(BasicEnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    @torch.no_grad()
    def forward_decoder(
        self, decoder_out, encoder_outs, eos_penalty=0.0, max_ratio=None, **kwargs
    ):
        # LevT ensembling
        # A pipeline of three steps: deletion, placeholder, and word insertion.
        # We need to average scores in each step in a pipeline way because of dependence.
        # deletion
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = output_tokens.new().fill_(255)
        else:
            if not encoder_outs[0]["encoder_padding_mask"]:
                src_lens = (
                    encoder_outs[0]["encoder_out"][0].new(bsz)
                    .fill_(encoder_outs[0]["encoder_out"][0].size(1))
                )
            else:
                src_lens = (~encoder_outs[0]["encoder_padding_mask"][0]).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            output_tokens, output_scores, attn = self.forward_word_del(
                encoder_outs,
                output_tokens,
                output_scores,
                attn,
                can_del_word,
            )

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            output_tokens, output_scores = self.forward_mask_ins(
                encoder_outs,
                output_tokens,
                output_scores,
                can_ins_mask,
                eos_penalty,
                max_lens,
            )

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            output_tokens, output_scores, attn = self.forward_word_ins(
                encoder_outs,
                output_tokens,
                output_scores,
                attn,
                can_ins_word,
            )

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=None,
        )

    def forward_word_del(
        self, encoder_outs, output_tokens, output_scores, attn, can_del_word
    ):
        word_del_score_avg = []
        word_del_attn_avg = []
        for model, encoder_out in zip(self.models, encoder_outs):
            word_del_out, word_del_attn = model.decoder.forward_word_del(
                _skip(output_tokens, can_del_word),
                _skip_encoder_out(model.encoder, encoder_out, can_del_word),
            )
            word_del_score = F.log_softmax(word_del_out, 2)
            word_del_score_avg.append(word_del_score)
            word_del_attn_avg.append(word_del_attn)
        word_del_score_avg = torch.logsumexp(
            torch.stack(word_del_score_avg, dim=0), dim=0
        ) - math.log(len(self.models))
        word_del_pred = word_del_score_avg.max(-1)[1].bool()
        if word_del_attn_avg[0] is not None:
            word_del_attn_avg = torch.stack(word_del_attn_avg, dim=0) / len(self.models)
        else:
            word_del_attn_avg = None

        _tokens, _scores, _attn = _apply_del_words(
            output_tokens[can_del_word],
            output_scores[can_del_word],
            word_del_attn_avg,
            word_del_pred,
            self.pad,
            self.bos,
            self.eos,
        )
        output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
        output_scores = _fill(output_scores, can_del_word, _scores, 0)
        attn = _fill(attn, can_del_word, _attn, 0.0)
        return output_tokens, output_scores, attn

    def forward_mask_ins(
        self,
        encoder_outs,
        output_tokens,
        output_scores,
        can_ins_mask,
        eos_penalty,
        max_lens,
    ):
        mask_ins_score_avg = []
        for model, encoder_out in zip(self.models, encoder_outs):
            mask_ins_out, _ = model.decoder.forward_mask_ins(
                _skip(output_tokens, can_ins_mask),
                _skip_encoder_out(model.encoder, encoder_out, can_ins_mask),
            )
            mask_ins_score = F.log_softmax(mask_ins_out, 2)
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] -= eos_penalty
            mask_ins_score_avg.append(mask_ins_score)
        mask_ins_score_avg = torch.logsumexp(
            torch.stack(mask_ins_score_avg, dim=0), dim=0
        ) - math.log(len(self.models))
        mask_ins_pred = mask_ins_score_avg.max(-1)[1]
        mask_ins_pred = torch.min(
            mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
        )
        _tokens, _scores = _apply_ins_masks(
            output_tokens[can_ins_mask],
            output_scores[can_ins_mask],
            mask_ins_pred,
            self.pad,
            self.unk,
            self.eos,
        )
        output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
        output_scores = _fill(output_scores, can_ins_mask, _scores, 0)
        return output_tokens, output_scores

    def forward_word_ins(
        self, encoder_outs, output_tokens, output_scores, attn, can_ins_word
    ):
        word_ins_score_avg = []
        word_ins_attn_avg = []
        for model, encoder_out in zip(self.models, encoder_outs):
            word_ins_out, word_ins_attn = model.decoder.forward_word_ins(
                _skip(output_tokens, can_ins_word),
                _skip_encoder_out(model.encoder, encoder_out, can_ins_word),
            )
            word_ins_score = F.log_softmax(word_ins_out, 2)
            word_ins_score_avg.append(word_ins_score)
            word_ins_attn_avg.append(word_ins_attn)
        word_ins_score_avg = torch.logsumexp(
            torch.stack(word_ins_score_avg, dim=0), dim=0
        ) - math.log(len(self.models))
        if word_ins_attn_avg[0] is not None:
            word_ins_attn_avg = torch.stack(word_ins_attn_avg, dim=0) / len(self.models)
        else:
            word_ins_attn_avg = None
        word_ins_score_max, word_ins_pred = word_ins_score_avg.max(-1)

        _tokens, _scores = _apply_ins_words(
            output_tokens[can_ins_word],
            output_scores[can_ins_word],
            word_ins_pred,
            word_ins_score_max,
            self.unk,
        )

        output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
        output_scores = _fill(output_scores, can_ins_word, _scores, 0)
        attn = _fill(attn, can_ins_word, word_ins_attn, 0.0)
        return output_tokens, output_scores, attn

    def initialize_output_tokens(self, encoder_outs, src_tokens):
        # LevT doesn't do length prediction.
        return self.models[0].initialize_output_tokens(encoder_outs[0], src_tokens)
