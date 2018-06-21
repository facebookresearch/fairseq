# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    FairseqDecoder, FairseqLanguageModel, register_model, register_model_architecture,
)

from fairseq import options

from fairseq.models.transformer import (
    Embedding, PositionalEmbedding, TransformerDecoderLayer
)

from fairseq.modules import (
    AdaptiveSoftmax,
)

from fairseq.utils import flip


@register_model('bi_transformer_lm')
class BiTransformerLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', default=0., type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', default=0., type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-fixed-pos', default=False, action='store_true',
                            help='use fixed positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_bi_lm_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.tokens_per_sample
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = args.tokens_per_sample

        embed_tokens = Embedding(len(task.dictionary), args.decoder_embed_dim, task.dictionary.pad())

        decoder = BiTransformerDecoder(args, task.dictionary, embed_tokens, no_encoder_attn=True)
        return BiTransformerLanguageModel(decoder)


class BiTransformerDecoder(FairseqDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.forward_layers = nn.ModuleList([TransformerDecoderLayer(args, no_encoder_attn)
                                             for _ in range(args.decoder_layers)])
        self.backward_layers = nn.ModuleList([TransformerDecoderLayer(args, no_encoder_attn)
                                              for _ in range(args.decoder_layers)])

        self.adaptive_softmax = None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary), args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.dropout
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self, source_tokens, encoder_out=None):
        # embed positions
        positions = self.embed_positions(source_tokens) if self.embed_positions is not None else None

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(source_tokens)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        rev_x = flip(x, 0)
        rev_attn = None
        rev_enc_out = flip(encoder_out['encoder_out'], 0) if encoder_out is not None else None
        rev_enc_pad = flip(encoder_out['encoder_padding_mask'], 0) if encoder_out is not None else None

        # decoder layers
        for fwd, back in zip(self.forward_layers, self.backward_layers):
            x, attn = fwd(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
            )
            rev_x, rev_attn = back(
                rev_x,
                rev_enc_out['encoder_out'] if rev_enc_out is not None else None,
                rev_enc_pad['encoder_padding_mask'] if rev_enc_pad is not None else None,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, attn

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        pass


@register_model_architecture('bi_transformer_lm', 'bi_transformer_lm')
def base_bi_lm_architecture(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = not getattr(args, 'decoder_fixed_pos', False)
    args.decoder_normalize_before = True
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
