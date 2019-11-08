#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.tracing_compliant_transformer import (
    TracingTransformerDecoder,
    TracingTransformerEncoder,
    TracingTransformerModel,
    TransformerDecoderLayer,
)
from fairseq.models.model_utils import (
    fill_tensors as _fill,
    script_skip_tensor,
    script_skip_tensor_list,
)
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from torch import Tensor


def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx):
    try:
        from fairseq import libnat
    except ImportError as e:
        import sys

        sys.stderr.write("ERROR: missing libnat. run `pip install --editable .`\n")
        raise e
    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

    in_tokens_list = [
        [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
    ]
    out_tokens_list = [
        [t for t in s if t != padding_idx] for i, s in enumerate(out_tokens.tolist())
    ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )
    mask_inputs = [
        [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
    ]

    # generate labels
    masked_tgt_masks = []
    for mask_input in mask_inputs:
        mask_label = []
        for beam_size in mask_input[1:-1]:  # HACK 1:-1
            mask_label += [0] + [1 for _ in range(beam_size)]
        masked_tgt_masks.append(
            mask_label + [0 for _ in range(out_seq_len - len(mask_label))]
        )
    mask_ins_targets = [
        mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
        for mask_input in mask_inputs
    ]

    # transform to tensor
    masked_tgt_masks = torch.tensor(masked_tgt_masks, device=out_tokens.device).bool()
    mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
    masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
    return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets


def _get_del_targets(in_tokens, out_tokens, padding_idx):
    try:
        from fairseq import libnat
    except ImportError as e:
        import sys

        sys.stderr.write("ERROR: missing libnat. run `pip install --editable .`\n")
        raise e
    out_seq_len = out_tokens.size(1)

    in_tokens_list = [
        [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
    ]
    out_tokens_list = [
        [t for t in s if t != padding_idx] for i, s in enumerate(out_tokens.tolist())
    ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )
    word_del_targets = [b[-1] for b in full_labels]
    word_del_targets = [
        labels + [0 for _ in range(out_seq_len - len(labels))]
        for labels in word_del_targets
    ]

    # transform to tensor
    word_del_targets = torch.tensor(word_del_targets)
    return word_del_targets


def _get_del_ins_targets(in_tokens, out_tokens, padding_idx):
    try:
        from fairseq import libnat
    except ImportError as e:
        import sys

        sys.stderr.write("ERROR: missing libnat. run `pip install --editable .`\n")
        raise e
    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

    in_tokens_list = [
        [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
    ]
    out_tokens_list = [
        [t for t in s if t != padding_idx] for i, s in enumerate(out_tokens.tolist())
    ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )

    word_del_targets = [b[-1] for b in full_labels]
    word_del_targets = [
        labels + [0 for _ in range(out_seq_len - len(labels))]
        for labels in word_del_targets
    ]

    mask_inputs = [
        [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
    ]
    mask_ins_targets = [
        mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
        for mask_input in mask_inputs
    ]

    # transform to tensor
    mask_ins_targets = torch.tensor(mask_ins_targets)
    word_del_targets = torch.tensor(word_del_targets)
    return word_del_targets, mask_ins_targets


@register_model("levenshtein_transformer")
class LevenshteinTransformerModel(TracingTransformerModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

    @staticmethod
    def add_args(parser):
        TracingTransformerModel.add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )
        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers for del_word, ins_mask, ins_word",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="addtional decoder-layers to learn deletion",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="addtional decoder-layers to learn predicting masks",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action="store_true",
            help="instead of argmax, use sampling to predict the tokens",
        )
        # Added for compatibility
        parser.add_argument(
            "--decoder-out-embed-dim",
            default=None,
            type=int,
            metavar="N",
            help="decoder output embedding dimension (bottleneck layer before"
            "output layer if specified.)",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LevenshteinTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TracingTransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            prev_output_tokens, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            prev_output_tokens, encoder_out=encoder_out
        )
        word_ins_out, _ = self.decoder.forward_word_ins(
            masked_tgt_tokens, encoder_out=encoder_out
        )

        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
            ).view(word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out, _ = self.decoder.forward_word_del(word_predictions, encoder_out)
        return {
            "mask_ins_out": mask_ins_out,
            "mask_ins_tgt": mask_ins_targets,
            "mask_ins_mask": mask_ins_masks,
            "word_ins_out": word_ins_out,
            "word_ins_tgt": tgt_tokens,
            "word_ins_mask": masked_tgt_masks,
            "word_del_out": word_del_out,
            "word_del_tgt": word_del_targets,
            "word_del_mask": word_predictions.ne(self.pad),
        }

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    def forward_decoder(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out[0]
        output_scores = decoder_out[1]
        attn = decoder_out[2]

        if max_ratio is not None and encoder_out[1] is not None:
            max_lengths = ((~encoder_out[1]).sum(1) * max_ratio).clamp(min=10)

        else:
            max_lengths = torch.zeros(output_tokens.size(0)).fill_(255)

        @torch.jit.script
        def del_word(
            output_tokens,
            output_scores,
            attn: Tensor,
            word_del_attn: Optional[Tensor],
            word_del_out,
            can_del_word,
            pad_idx: int,
            bos_idx: int,
            eos_idx: int,
        ):
            # delete words
            # do not delete tokens if it is <s> </s>
            if can_del_word.sum() != 0:  # we cannot delete, skip
                word_del_score = F.log_softmax(word_del_out, 2)
                word_del_pred = torch.jit.Attribute(word_del_score.max(-1)[1], bool)
                in_tokens = output_tokens[can_del_word]
                in_scores = output_scores[can_del_word]
                # apply deletion to a tensor
                in_masks = in_tokens.ne(pad_idx)
                bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

                max_len = in_tokens.size(1)
                word_del_pred.masked_fill_(~in_masks, 1)
                word_del_pred.masked_fill_(bos_eos_masks, 0)

                reordering = (
                    torch.arange(max_len)[None, :]
                    .expand_as(in_tokens)
                    .contiguous()
                    .masked_fill(word_del_pred, max_len)
                    .sort(1)[1]
                )

                _tokens = in_tokens.masked_fill(word_del_pred, pad_idx).gather(
                    1, reordering
                )

                _scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)
                if word_del_attn is not None:
                    _mask = word_del_pred[:, :, None].expand_as(word_del_attn)
                    _reordering = reordering[:, :, None].expand_as(word_del_attn)
                    _attn = word_del_attn.masked_fill(_mask, 0.0).gather(1, _reordering)
                    attn = _fill(attn, can_del_word, _attn, 0)

                output_tokens = _fill(output_tokens, can_del_word, _tokens, pad_idx)
                output_scores = _fill(output_scores, can_del_word, _scores, 0)
            return output_tokens, output_scores, attn

        @torch.jit.script
        def ins_placeholders(
            output_tokens,
            output_scores,
            mask_ins_out,
            can_ins_mask,
            pad_idx: int,
            unk_idx: int,
            eos_idx: int,
            max_ratio: float,
            max_lengths,
        ):
            # insert placeholders
            if can_ins_mask.sum() != 0:
                mask_ins_score = F.log_softmax(mask_ins_out, 2)
                if eos_penalty > 0.0:
                    mask_ins_score[:, :, 0] -= eos_penalty
                mask_ins_pred = mask_ins_score.max(-1)[1]
                if max_ratio is not None and encoder_out[1] is not None:
                    mask_ins_pred = torch.min(
                        mask_ins_pred, max_lengths[can_ins_mask, None].expand_as(mask_ins_pred)
                    )
                in_tokens = output_tokens[can_ins_mask]
                in_scores = output_scores[can_ins_mask]
                in_masks = in_tokens.ne(pad_idx)
                in_lengths = in_masks.sum(1)

                # HACK: hacky way to shift all the paddings to eos first.
                in_tokens.masked_fill_(~in_masks, eos_idx)
                mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

                out_lengths = in_lengths + mask_ins_pred.sum(1)
                out_max_len = out_lengths.max()
                out_masks = (
                    torch.arange(out_max_len)[None, :].long() < out_lengths[:, None]
                )

                reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
                out_tokens = (
                    torch.zeros(in_tokens.size()[0], out_max_len)
                    .fill_(pad_idx)
                    .masked_fill_(out_masks, unk_idx)
                )
                out_tokens = torch.cat([in_tokens[:, :1], out_tokens[:, 1:]], 1)
                out_tokens.scatter_(1, reordering, in_tokens[:, 1:].float())

                if in_scores is not None:
                    in_scores.masked_fill_(~in_masks, 0)
                    out_scores = torch.zeros_like(out_tokens).to(in_scores)
                    out_tokens = torch.cat([in_tokens[:, :1], out_tokens[:, 1:]], 1)
                    out_scores.scatter_(1, reordering, in_scores[:, 1:])
                else:
                    out_scores = None
                output_tokens = _fill(output_tokens, can_ins_mask, out_tokens, pad_idx)
                output_scores = _fill(output_scores, can_ins_mask, out_scores, 0)
            return output_tokens, output_scores

        @torch.jit.script
        def ins_words(
            output_tokens,
            output_scores,
            attn: Tensor,
            word_ins_attn,
            word_ins_out,
            can_ins_word,
            pad_idx: int,
            unk_idx: int,
        ):
            # insert words
            if can_ins_word.sum() != 0:
                word_ins_scores = F.log_softmax(word_ins_out, 2)
                word_ins_pred = word_ins_scores.max(-1)[1]
                in_tokens = output_tokens[can_ins_word]
                in_scores = output_scores[can_ins_word]
                word_ins_masks = in_tokens.eq(unk_idx)
                out_tokens = in_tokens.masked_scatter(
                    word_ins_masks, word_ins_pred[word_ins_masks].float()
                )

                if in_scores is not None:
                    out_scores = in_scores.masked_scatter(
                        word_ins_masks, word_ins_scores[word_ins_masks]
                    )
                else:
                    out_scores = None
                output_tokens = _fill(output_tokens, can_ins_word, out_tokens, pad_idx)
                output_scores = _fill(output_scores, can_ins_word, out_scores, 0)
                attn = _fill(attn, can_ins_word, word_ins_attn, 0)
            return output_tokens, output_scores, attn

        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        word_del_out, word_del_attn = self.decoder.forward_word_del(
            script_skip_tensor(output_tokens, can_del_word),
            script_skip_tensor_list(list(encoder_out), can_del_word),
        )

        output_tokens, output_scores, attn = del_word(
            output_tokens,
            output_scores,
            attn,
            word_del_attn,
            word_del_out,
            can_del_word,
            self.pad,
            self.bos,
            self.eos,
        )

        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lengths
        mask_ins_out, _ = self.decoder.forward_mask_ins(
            script_skip_tensor(output_tokens, can_ins_mask),
            script_skip_tensor_list(encoder_out, can_ins_mask),
        )
        output_tokens, output_scores = ins_placeholders(
            output_tokens,
            output_scores,
            mask_ins_out,
            can_ins_mask,
            self.pad,
            self.unk,
            self.eos,
            max_ratio,
            max_lengths,
        )

        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        word_ins_out, word_ins_attn = self.decoder.forward_word_ins(
            script_skip_tensor(output_tokens, can_ins_word),
            script_skip_tensor_list(encoder_out, can_ins_word),
        )


        output_tokens, output_scores, attn = ins_words(
            output_tokens,
            output_scores,
            attn,
            word_ins_attn,
            word_ins_out,
            can_ins_word,
            self.pad,
            self.unk,
        )

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()

        @torch.jit.script
        def slice_wrap(x, l):
            return x[:, :l]

        @torch.jit.script
        def slice_wrap_attn(x, l):
            return x if x.size()[0] == 0 else x[:, :l, :]

        output_tokens = slice_wrap(output_tokens, cut_off)
        output_scores = slice_wrap(output_scores, cut_off)
        attn = slice_wrap(attn, cut_off)
        return [output_tokens, output_scores, attn, 0, 0]

    def initialize_output_tokens(self, encoder_out, src_tokens):
        initial_output_tokens = torch.cat(
            [
                torch.zeros(src_tokens.size(0), 1).fill_(self.bos),
                torch.zeros(src_tokens.size(0), 1).fill_(self.eos),
            ],
            1,
        )

        initial_output_scores = torch.zeros_like(initial_output_tokens).to(
            encoder_out[0]
        )

        initial_attn = torch.empty([0])
        if getattr(self.decoder.layers[-1], "need_attn", True):
            initial_attn = torch.zeros([src_tokens.size(0), 2, src_tokens.size(1)]).to(
                initial_output_tokens
            )

        return [initial_output_tokens, initial_output_scores, initial_attn, 0, 0]


class LevenshteinTransformerDecoder(TracingTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(",")]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[1])
                ]
            )
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[0])
                ]
            )

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        layers=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens.long())
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn = layer(
                x,
                encoder_out[0] if encoder_out is not None else None,
                encoder_out[1] if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, attn, inner_states

    def forward_mask_ins(self, prev_output_tokens, encoder_out=None, **unused):
        features, attn, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[1],
            layers=self.layers_msk,
            **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        return F.linear(features_cat, self.embed_mask_ins.weight), attn

    def forward_word_ins(self, prev_output_tokens, encoder_out=None, **unused):
        features, attn, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        return self.output_layer(features), attn

    def forward_word_del(self, prev_output_tokens, encoder_out=None, **unused):
        features, attn, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            **unused
        )
        return F.linear(features, self.embed_word_del.weight), attn


@register_model_architecture("levenshtein_transformer", "levenshtein_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)


@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de"
)
def levenshtein_transformer_wmt_en_de(args):
    base_architecture(args)


# similar parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_vaswani_wmt_en_de_big"
)
def levenshtein_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de_big"
)
def levenshtein_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    levenshtein_transformer_vaswani_wmt_en_de_big(args)
