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
from fairseq import utils

from fairseq.models.transformer import (
    Embedding, LayerNorm, Linear, PositionalEmbedding,
)

from fairseq.modules import (
    AdaptiveSoftmax, BidirectionalMultiheadSelfAttention, CharacterTokenEmbedder, MultiheadAttention,
    SinusoidalPositionalEmbedding
)


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
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', type=int, metavar='N', default=4,
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', type=int, metavar='N', default=2,
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--exclude-self-target', action='store_true',
                            help='exclude self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        parser.add_argument('--linear-final-layer', action='store_true',
                            help='if set, uses a simple linear layer for the final prediction that combines the '
                                 'forward and backward tower instead of an attentional layer')
        parser.add_argument('--linear-final-layer-bias', action='store_true',
                            help='if set, has a bias on the final linear layer')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_bi_lm_architecture(args)

        cls.set_targets(args, task)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.tokens_per_sample
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = args.tokens_per_sample

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(task.dictionary, eval(args.character_filters),
                                                  args.character_embedding_dim,
                                                  args.decoder_embed_dim,
                                                  args.char_embedder_highway_layers,
                                                  )
        else:
            embed_tokens = Embedding(len(task.dictionary), args.decoder_embed_dim, task.dictionary.pad())

        print("Model args: ", args)

        decoder = BiTransformerDecoder(args, task.dictionary, embed_tokens)
        return BiTransformerLanguageModel(decoder)

    @classmethod
    def set_targets(cls, args, task):
        targets = ['self'] if not args.exclude_self_target else []
        if args.future_target:
            targets.append('future')
        if args.past_target:
            targets.append('past')

        for ds in task.datasets.values():
            ds.set_targets(targets)

    def remove_head(self):
        self.decoder.remove_head()


class BiTransformerDecoder(FairseqDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.exclude_self_target = args.exclude_self_target
        self.future_target = args.future_target
        self.past_target = args.past_target

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.forward_layers = nn.ModuleList([TransformerDecoderLayer(args)
                                             for _ in range(args.decoder_layers)])
        self.backward_layers = nn.ModuleList([TransformerDecoderLayer(args)
                                              for _ in range(args.decoder_layers)])

        self.full_attn_layer = None
        self.full_linear_layer = None

        if not self.exclude_self_target:
            if args.linear_final_layer:
                self.full_linear_layer = Linear(embed_dim * 2, embed_dim, args.linear_final_layer_bias)
            else:
                self.full_attn_layer = BidirectionalTransformerDecoderLayer(args)

        self.embed_out = None
        self.adaptive_softmax = None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary), args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self, source_tokens, **unused):
        """ Forward pass for the bidirectional transformer

        Args:
            - source tokens: B x T matrix representing sentences

        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in softmax afterwards
                - a dictionary of additional data, where 'attn' contains the attention over the final
                  states (concatenated from forward and backward towers) and 'inner_states' is a list
                  of internal model states used to compute the predictions (for example to use in ELMO).
                  The first element is the token embeddings (with the positional embeddings added).
                  The next n elements are tuples of the hidden states for the forward and backward towers.
                  The last element is the output of the final full layer on top of the towers and would be
                  equivalent to the logits if adaptive softmax is used.
                  NOTE: unlike the logits, the format for all hidden states is T x B x C
        """

        # compute padding mask
        padding_mask = source_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        # embed positions
        positions = self.embed_positions(source_tokens) if self.embed_positions is not None else None

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(source_tokens)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        fwd_x = bwd_x = x.transpose(0, 1)

        inner_states = [fwd_x]

        future_mask = self.buffered_future_mask(fwd_x)
        past_mask = self.buffered_past_mask(bwd_x)

        # decoder layers
        for fwd, back in zip(self.forward_layers, self.backward_layers):
            fwd_x, _ = fwd(
                fwd_x,
                self_attn_mask=future_mask,
                self_attn_padding_mask=padding_mask,
            )
            bwd_x, _ = back(
                bwd_x,
                self_attn_mask=past_mask,
                self_attn_padding_mask=padding_mask,
            )
            inner_states.extend((fwd_x, bwd_x))

        if not self.exclude_self_target:
            if self.full_attn_layer is not None:
                x, attn = self.full_attn_layer(
                    fwd_x,
                    bwd_x,
                    padding_mask,
                )
                inner_states.append(x)
            elif self.full_linear_layer is not None:
                zeros = x.new_zeros(1, fwd_x.size(1), fwd_x.size(2))
                fwd_x = torch.cat([zeros, fwd_x[:-1]], dim=0)
                bwd_x = torch.cat([bwd_x[1:], zeros], dim=0)
                x = torch.cat([fwd_x, bwd_x], dim=-1)
                x = self.full_linear_layer(x)
                attn = None
                inner_states.append(x)
            x = [x]
        else:
            x = []
            attn = None

        if self.future_target:
            x.append(fwd_x)
        if self.past_target:
            x.append(bwd_x)

        # T x B x C -> B x T x C
        x = [z.transpose(0, 1) for z in x]

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed and hasattr(self.embed_tokens, 'weight'):
                x = [F.linear(x, self.embed_tokens.weight) for x in x]
            elif self.embed_out is not None:
                x = [F.linear(x, self.embed_out) for x in x]

        if len(x) == 1:
            x = x[0]

        return x, {'attn': attn, 'inner_states': inner_states}

    def remove_head(self):
        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            del self.adaptive_softmax
            self.adaptive_softmax = None
        if hasattr(self, 'embed_out') and self.embed_out is not None:
            del self.embed_out
            self.embed_out = None

        self.share_input_output_embed = False

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def buffered_past_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_past_mask') or self._past_mask is None or self._past_mask.device != tensor.device:
            self._past_mask = torch.tril(utils.fill_with_neg_inf(tensor.new(dim, dim)), -1)
        if self._past_mask.size(0) < dim:
            self._past_mask = torch.tril(utils.fill_with_neg_inf(self._past_mask.resize_(dim, dim)), -1)
        return self._past_mask[:dim, :dim]

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor(1)
        return state_dict


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout, add_bias_kv=True,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class BidirectionalTransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = BidirectionalMultiheadSelfAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.fwd_layer_norm = LayerNorm(self.embed_dim)
        self.bwd_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, fwd_x, bwd_x, key_padding_mask):
        fwd_x = self.maybe_layer_norm(self.fwd_layer_norm, fwd_x, before=True)
        bwd_x = self.maybe_layer_norm(self.bwd_layer_norm, bwd_x, before=True)
        x, attn = self.self_attn(
            fwd_x=fwd_x,
            bwd_x=bwd_x,
            key_padding_mask=key_padding_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.maybe_layer_norm(self.fwd_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


@register_model_architecture('bi_transformer_lm', 'bi_transformer_lm')
def base_bi_lm_architecture(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.character_embeddings = getattr(args, 'character_embeddings', False)
    args.character_filters = getattr(args, 'character_filters',
                                     '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]')
    args.character_embedding_dim = getattr(args, 'character_embedding_dim', 128)
    args.char_embedder_highway_layers = getattr(args, 'char_embedder_highway_layers', 2)
    args.linear_final_layer = getattr(args, 'linear_final_layer', False)
    args.linear_final_layer_bias = getattr(args, 'linear_final_layer_bias', False)

    args.exclude_self_target = getattr(args, 'exclude_self_target', False)
    args.future_target = getattr(args, 'future_target', False)
    args.past_target = getattr(args, 'past_target', False)

    # otherwise model training is unstable
    args.decoder_normalize_before = True


@register_model_architecture('bi_transformer_lm', 'bi_transformer_lm_big')
def bi_transformer_lm_big(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_bi_lm_architecture(args)
