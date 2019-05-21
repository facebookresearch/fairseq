#!/usr/bin/env python
# encoding: utf-8
"""
@author: Wang Qiang
@contact: wangqiangneu@gmail.com
@desc: The implementation of "Learning Deep Transformer Models for Machine Translation"
"""


import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.dlcl import DynamicLinearCombination
from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention, MultiheadAttentionV2,
    SinusoidalPositionalEmbedding,
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel,
    register_model, register_model_architecture,
)

import fairseq.utils as util

_MODEL_NAME_ = "dlcl_transformer"


@register_model(_MODEL_NAME_)
class DlclTransformerModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--inspect-grad', type=eval, default='False',
                            help='inspect intermediate gradient in tensorboard. Use more GPU memory')
        # DLCL parameters
        parser.add_argument('--init-value', type=str, default='avg', choices=['avg', 'one'],
                            help='how to init the learned weight matrix')
        parser.add_argument('--weight-type', type=str, default='scalar',
                            help='type of learned weight [scalar, scalar_n(n>1), vector]')
        parser.add_argument('--encoder-learnable', type=eval, default='True',
                            help='enable to learn weights for encoder')
        parser.add_argument('--decoder-learnable', type=eval, default='True',
                            help='enable to learn weights for decoder')
        parser.add_argument('--normalize-learned-weight', type=eval, default='False',
                            help='normalize learned weight by softmax')
        parser.add_argument('--normalize-embedding', type=eval, default='False',
                            help='normalize the input of embedding')
        parser.add_argument('--history-dropout', type=float, default=0.0, metavar='D',
                            help='dropout for history output')
        parser.add_argument('--history-window-size', type=int, default='-1',
                            help='how many past layers are considered. -1 means all')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = DlclTransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = DlclTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return DlclTransformerModel(encoder, decoder)


class DlclTransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.history = DynamicLinearCombination(args, is_encoder=True)
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        self.inspected_grads = OrderedDict() if getattr(args, 'inspect_grad', False) else None

    def forward(self, src_tokens, src_lengths):
        # clean layer history
        self.history.clean()

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        util.inspect_grad("encoder_0", x, self.inspected_grads)

        # push embedding layer into memory
        self.history.push(x)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer_id, layer in enumerate(self.layers):
            # fetch combined input from memory for the next layer
            x = self.history.pop()
            x = layer(x, encoder_padding_mask)
            # push into memory
            self.history.push(x)
            util.inspect_grad("encoder_%d" % (layer_id + 1), x, self.inspected_grads)

        # read from memory
        x = self.history.pop()
        if self.normalize:
            x = self.layer_norm(x)
        util.inspect_grad("encoder_top", x, self.inspected_grads)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            if 'encoder.embed_positions._float_tensor' not in state_dict:
                state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class DlclTransformerDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for i in range(args.decoder_layers)
        ])
        self.history = DynamicLinearCombination(args, is_encoder=False)
        self.normalize = args.decoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

        self.inspected_grads = OrderedDict() if getattr(args, 'inspect_grad', False) else None

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        self.history.clean()

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        util.inspect_grad('decoder_0', x, self.inspected_grads)
        # push embedding layer into memory
        self.history.push(x)

        # decoder layers
        for layer_id, layer in enumerate(self.layers):
            # read from memory
            x = self.history.pop()
            x, attn = layer(
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                incremental_state,
            )
            # write into memory
            self.history.push(x)
            util.inspect_grad('decoder_%d' % (layer_id + 1), x, self.inspected_grads)

        # read from memory
        x = self.history.pop()
        if self.normalize:
            x = self.layer_norm(x)
        util.inspect_grad('decoder_top', x, self.inspected_grads)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

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
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before
        if self.embed_dim == args.encoder_embed_dim:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
        else:
            # support different dimension between encoder and decoder
            self.encoder_attn = MultiheadAttentionV2(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
                kv_dim=args.encoder_embed_dim
            )
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, init_size=num_embeddings)
    return m


@register_model_architecture(_MODEL_NAME_, _MODEL_NAME_)
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.inspect_grad = getattr(args, 'inspect_grad', False)

    # setting for DLCL
    args.init_value = getattr(args, 'init_value', 'avg')
    args.weight_type = getattr(args, 'weight_type', 'scalar')
    args.encoder_learnable = getattr(args, 'encoder_learnable', True)
    args.decoder_learnable = getattr(args, 'decoder_learnable', True)
    args.normalize_embed = getattr(args, 'normalize_embed', False)
    args.history_dropout = getattr(args, 'history_dropout', 0.0)
    args.history_window_size = getattr(args, 'history_window_size', -1)
    

@register_model_architecture(_MODEL_NAME_, '%s_postnorm_wmt_en_de' % _MODEL_NAME_)
def dlcl_transformer_postnorm_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture(_MODEL_NAME_, '%s_postnorm_deep_wmt_en_de' % _MODEL_NAME_)
def dlcl_transformer_postnorm_deep_wmt_en_de(args):
    args.encoder_layers = 25
    dlcl_transformer_postnorm_wmt_en_de(args)


@register_model_architecture(_MODEL_NAME_, '%s_prenorm_wmt_en_de' % _MODEL_NAME_)
def dlcl_transformer_prenorm_wmt_en_de(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    base_architecture(args)


@register_model_architecture(_MODEL_NAME_, '%s_prenorm_deep_wmt_en_de' %_MODEL_NAME_)
def dlcl_transformer_prenorm_deep_wmt_en_de(args):
    args.encoder_layers = 30
    dlcl_transformer_prenorm_wmt_en_de(args)


@register_model_architecture(_MODEL_NAME_, '%s_iwslt_de_en' %_MODEL_NAME_)
def dlcl_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    base_architecture(args)


@register_model_architecture(_MODEL_NAME_, '%s_prenorm_iwslt_de_en' %_MODEL_NAME_)
def dlcl_transformer_prenorm_iwslt_de_en(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    dlcl_transformer_iwslt_de_en(args)


@register_model_architecture(_MODEL_NAME_, '%s_toy'%_MODEL_NAME_)
def dlcl_transformer_toy(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 64)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 64)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    base_architecture(args)
