# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
# Original Authors:
# Mattia A. Di Gangi, Matteo Negri, Roldano Cattoni, Roberto Dessi, Marco Turchi
#
# Original source:
# https://github.com/mattiadg/FBK-Fairseq-ST
#
# Reference:
# Enhancing Transformer for End-to-end Speech-to-Text Translation
# at MT Summit XVII 2020


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils

from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)

from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
)

from fairseq.models.fairseq_encoder import (
    EncoderOut,
)

from fairseq.modules import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    TransformerEncoderLayer,
)

from .modules.multihead_attention import (
    LocalMultiheadAttention,
    ConvAttention2D,
)


@register_model('speechconvtransformer')
class SpeechTransformerModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(SpeechTransformerModel, SpeechTransformerModel).add_args(parser)
        parser.add_argument('--encoder-convolutions', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--normalization-constant',
                            type=float, default=1.0)
        parser.add_argument('--distance-penalty', type=str, default=False,
                            choices=['log', 'gauss'],
                            help='Add distance penalty to the encoder')
        parser.add_argument('--init-variance', type=float, default=1.0,
                            help='Initialization value for variance')
        parser.add_argument("--input-feat-per-channel", type=int, metavar="N",
                            help=(
                                "Encoder input dimension per input channel. "
                                "Typical values for speech are 40 or 80."),)

        parser.add_argument('--no-attn-2d', action='store_false',
                            dest='attn_2d', help="Don't use 2d attention")
        parser.add_argument('--attn-2d', action='store_true', dest='attn_2d',
                            help="Use 2d attention")
        parser.set_defaults(attn_2d=True)

        parser.add_argument('--no-batch-norm', action="store_false",
                            dest='batch_norm', help='No batch norm layer')
        parser.add_argument('--batch-norm', action="store_true",
                            dest='batch_norm', help='Use batch norm layer')
        parser.set_defaults(batch_norm=False)

        parser.add_argument("--load-pretrained-encoder-from", type=str,
                            metavar="STR", help=(
                                "model to take encoder"
                                "weights from (for initialization)"))
        parser.add_argument("--encoder-unidirectional", default=False, action="store_true",
                            help="Unidirectional encoder",)

    @classmethod
    def build_model(cls, args, task):
        encoder = cls.build_encoder(args)

        tgt_dict = task.target_dictionary
        decoder_embed_tokens = cls.build_embedding(
            args, tgt_dict, args.decoder_embed_dim
        )

        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, *extra_args):
        encoder = SpeechTransformerEncoder(args)

        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder


class SpeechTransformerEncoder(TransformerEncoder):
    """Transformer encoder."""

    def __init__(
        self,
        args,
        padding_idx=0,
        left_pad=False,
        convolutions=((512, 3),) * 20, stride=2,
    ):
        FairseqEncoder.__init__(self, None)
        self.padding_idx = padding_idx
        self.dropout = args.dropout
        self.max_source_positions = args.max_source_positions
        self.input_feat_per_channel = args.input_feat_per_channel

        if args.encoder_convolutions is not None:
            convolutions = eval(args.encoder_convolutions)

        convolutions = extend_conv_spec(convolutions)

        self.convolutions = nn.ModuleList()

        in_channels = 1
        for (
            i, (out_channels, kernel_size, kernel_width)
        ) in enumerate(convolutions):
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0

            self.convolutions.append(
                Conv2D(in_channels, out_channels, kernel_size,
                       dropout=self.dropout, padding=padding, stride=2)
            )
            in_channels = out_channels

        self.relu = nn.ReLU()

        self.batch_norm = getattr(args, 'batch_norm', False)

        self.encoder_unidiractional = args.encoder_unidirectional

        if args.attn_2d:
            self.attn_2d = nn.ModuleList(
                [
                    ConvAttention2D(
                        embed_dim=out_channels,
                        num_heads=4,
                        dropout=self.dropout,
                        batch_norm=self.batch_norm,
                        unidirectional=self.encoder_unidiractional,
                    ) for _ in range(2)
                ],
            )
        else:
            self.attn_2d = list()

        if self.batch_norm:
            self.bn = nn.ModuleList(
                [
                    BatchNorm(out_channels)
                    for _ in range(len(convolutions))
                ]
            )
        else:
            self.bn = None

        flat_dim = math.ceil(
            math.ceil(self.input_feat_per_channel / 2) / 2
        ) * out_channels

        self.layers = nn.ModuleList([])

        self.fc3 = Linear(flat_dim, args.encoder_embed_dim)
        self.layers.extend([
            SpeechTransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        self.embed_positions = PositionalEmbeddingAudio(
            args.max_source_positions, args.encoder_embed_dim,
            self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)

        # TODO: what's this?
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, src_tokens, src_lengths, **extra_args):
        # B x T x C -> B x 1 x T x C
        x = src_tokens.unsqueeze(1)
        # temporal convolutions
        for i, conv in enumerate(self.convolutions):
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = self.relu(x)
            if self.bn is not None:
                x = self.bn[i](x)
            src_lengths = torch.ceil(src_lengths.float() / 2).long()
            x = F.dropout(x, p=max(self.dropout, .1), training=self.training)

        encoder_padding_mask = self.create_mask(src_lengths)

        for layer in self.attn_2d:
            residual = x
            x, _ = layer(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
            x = x + residual

        bsz, out_channels, time, feats = x.size()
        x = (
            x
            .transpose(1, 2)
            .contiguous()
            .view(bsz, time, -1)
            .contiguous()
            .transpose(0, 1)
        )

        x = self.relu(self.fc3(x))

        x = x + self.embed_positions(
            x.transpose(0, 1),
            encoder_padding_mask
        ).transpose(0, 1)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.encoder_unidiractional:
            seq_len, _, _ = x.size()
            attn_mask = x.new_ones([seq_len, seq_len]).triu(1)
        else:
            attn_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask, attn_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_lengths=src_tokens,
            src_tokens=src_lengths,
        )

    def create_mask(self, lengths):
        max_len = max(lengths)
        mask = lengths.new_zeros(len(lengths), max_len).byte()
        for i, l in enumerate(lengths):
            mask[i, l:] = 1
        if not mask.any():
            mask = None
        return mask

    def upgrade_state_dict_named(self, state_dict, *args):
        return state_dict

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            state_dict['encoder.embed_positions._float_tensor'] = \
                torch.FloatTensor(1)
        if state_dict.get('encoder.version', torch.Tensor([2]))[0].float() < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['encoder.version'] = torch.Tensor([1])
        return state_dict


class SpeechTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        if args.distance_penalty is not False:
            self.self_attn = LocalMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout, penalty=args.distance_penalty,
                init_variance=(
                    args.init_variance
                    if args.distance_penalty == 'gauss' else None
                ),
                batch_norm=getattr(args, "batch_norm", False)
            )


def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally
    how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception(
                f'invalid number of parameters in convolution spec f{spec}.'
                f'expected 2 or 3'
            )
    return tuple(extended)


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def Conv2D(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv2d layer"""
    m = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return m


def BatchNorm(embedding_dim):
    m = nn.BatchNorm2d(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m


class PositionalEmbeddingAudio(nn.Module):
    """This module learns audio positional embeddings up to a fixed maximum size.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(
        self, num_embeddings, embedding_dim,
        padding_idx, learned=True
    ):
        super().__init__()
        if learned:
            self.embeddings = LearnedPositionalEmbedding(
                num_embeddings, embedding_dim, padding_idx
            )
        else:
            self.embeddings = SinusoidalPositionalEmbedding(
                embedding_dim, padding_idx
            )
        self.padding_idx = padding_idx

    def forward(
        self,
        input,
        encoder_padding_mask=None,
        incremental_state=None
    ):
        """Input is expected to be of size [bsz x seqlen x feature_dim]."""
        pos_tensor = input.new(
            input.size(0), input.size(1)
        ).fill_(self.padding_idx + 1)

        if encoder_padding_mask is not None:
            pos_tensor = pos_tensor.masked_fill(
                encoder_padding_mask.bool(),
                self.padding_idx
            )

        return self.embeddings(pos_tensor)

    @property
    def max_positions(self):
        """Maximum number of supported positions."""
        return self.embeddings.max_positions

    @property
    def weight(self):
        return self.embeddings.weight


@register_model_architecture(
    'speechconvtransformer',
    'speechconvtransformer'
)
def speechtransformer_base(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.normalization_constant = getattr(args, 'normalization_constant', 1.0)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.attn_2d = not getattr(args, 'no_attn_2d', False)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_convolutions = getattr(
        args, 'encoder_convolutions', '[(64, 3, 3)] * 2')
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_normalize_before = getattr(
        args, 'encoder_normalize_before', True)
    args.distance_penalty = getattr(args, 'distance_penalty', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_normalize_before = getattr(
        args, 'decoder_normalize_before', True)

    args.max_source_positions = getattr(args, "maxx_source_positions", 3000)
    args.max_target_positions = getattr(args, "maxx_target_positions", 1024)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False)
    args.no_scale_embedding = getattr(
        args, "no_scale_embedding", False)
    args.adaptive_input = getattr(
        args, "adaptive_input", False)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
