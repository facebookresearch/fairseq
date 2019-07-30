# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq.models import (
    CompositeEncoder,
    FairseqDecoder,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    DownsampledMultiHeadAttention,
    GradMultiply,
    LayerNorm,
    LearnedPositionalEmbedding,
    LinearizedConvolution,
)


@register_model('fconv_self_att')
class FConvModelSelfAtt(FairseqEncoderDecoderModel):

    @classmethod
    def hub_models(cls):
        return {
            'conv.stories': 'https://dl.fbaipublicfiles.com/fairseq/models/stories_checkpoint.tar.bz2',
            # Test set containing dictionaries
            'data.stories': 'https://dl.fbaipublicfiles.com/fairseq/data/stories_test.tar.bz2',
        }

    def __init__(self, encoder, decoder, pretrained_encoder=None):
        super().__init__(encoder, decoder)
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)
        self.pretrained_encoder = pretrained_encoder
        if self.pretrained_encoder is None:
            encoders = {'encoder': encoder}
        else:
            encoders = {'encoder': encoder, 'pretrained': self.pretrained_encoder}
        # for fusion model, CompositeEncoder contains both pretrained and training encoders
        # these are forwarded and then combined in the decoder
        self.encoder = CompositeEncoder(encoders)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')
        parser.add_argument('--self-attention', type=str, metavar='EXPR',
                            help='decoder self-attention layers, ex: [True] + [False]*5')
        parser.add_argument('--multihead-attention-nheads', type=int,
                            help='Number of heads to use in attention')
        parser.add_argument('--multihead-self-attention-nheads', type=int,
                            help='Number of heads to use in self-attention')
        parser.add_argument('--encoder-attention', type=str, metavar='EXPR',
                            help='encoder attention [True, ...]')
        parser.add_argument('--encoder-attention-nheads', type=int,
                            help='Number of heads to use in encoder attention')
        parser.add_argument('--project-input', type=str, metavar='EXPR',
                            help='Use projections in self-attention [True, ...]')
        parser.add_argument('--gated-attention', type=str, metavar='EXPR',
                            help='Use GLU layers in self-attention projections [True, ...]')
        parser.add_argument('--downsample', type=str, metavar='EXPR',
                            help='Use downsampling in self-attention [True, ...]')
        parser.add_argument('--pretrained-checkpoint', metavar='DIR',
                            help='path to load checkpoint from pretrained model')
        parser.add_argument('--pretrained', type=str, metavar='EXPR',
                            help='use pretrained model when training [True, ...]')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        trained_encoder, trained_decoder = None, None
        pretrained = eval(args.pretrained)
        if pretrained:
            print("| loading pretrained model")
            trained_model = checkpoint_utils.load_model_ensemble(
                filenames=[args.pretrained_checkpoint],
                task=task,
            )[0][0]
            trained_decoder = list(trained_model.children())[1]
            trained_encoder = list(trained_model.children())[0]

            # freeze pretrained model
            for param in trained_decoder.parameters():
                param.requires_grad = False
            for param in trained_encoder.parameters():
                param.requires_grad = False

        encoder = FConvEncoder(
            task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
            attention=eval(args.encoder_attention),
            attention_nheads=args.encoder_attention_nheads
        )

        decoder = FConvDecoder(
            task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            selfattention=eval(args.self_attention),
            attention_nheads=args.multihead_attention_nheads,
            selfattention_nheads=args.multihead_self_attention_nheads,
            project_input=eval(args.project_input),
            gated_attention=eval(args.gated_attention),
            downsample=eval(args.downsample),
            pretrained=pretrained,
            trained_decoder=trained_decoder
        )
        model = FConvModelSelfAtt(encoder, decoder, trained_encoder)

        return model

    @property
    def pretrained(self):
        return self.pretrained_encoder is not None


class FConvEncoder(FairseqEncoder):
    """Convolutional encoder"""
    def __init__(
        self, dictionary, embed_dim=512, max_positions=1024,
        convolutions=((512, 3),) * 20, dropout=0.1, attention=False,
        attention_nheads=1,
    ):
        super().__init__(dictionary)
        self.dropout = dropout
        self.num_attention_layers = None

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
        )

        def expand_bool_array(val):
            if isinstance(val, bool):
                # expand True into [True, True, ...] and do the same with False
                return [val] * len(convolutions)
            return val

        attention = expand_bool_array(attention)

        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.attproj = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            self.projections.append(
                Linear(in_channels, out_channels) if in_channels != out_channels else None
            )
            self.convolutions.append(
                ConvTBC(in_channels, out_channels * 2, kernel_size, dropout=dropout)
            )

            self.attention.append(
                SelfAttention(out_channels, embed_dim, attention_nheads) if attention[i] else None
            )
            in_channels = out_channels

        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x.transpose(0, 1)

        # project to size of convolution
        x = self.fc1(x)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            residual = x if proj is None else proj(x)

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = F.dropout(x, p=self.dropout, training=self.training)
            padding_l = (conv.kernel_size[0] - 1) // 2
            padding_r = conv.kernel_size[0] // 2
            x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
            x = conv(x)
            x = F.glu(x, dim=2)
            if attention is not None:
                x = attention(x)
            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding.transpose(0, 1)) * math.sqrt(0.5)

        return {
            'encoder_out': (x, y),
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(0, new_order) for eo in encoder_out['encoder_out']
        )

        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)

        if 'pretrained' in encoder_out:
            encoder_out['pretrained']['encoder_out'] = tuple(
                eo.index_select(0, new_order)
                for eo in encoder_out['pretrained']['encoder_out']
            )

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()


class FConvDecoder(FairseqDecoder):
    """Convolutional decoder"""
    def __init__(
        self, dictionary, embed_dim=512, out_embed_dim=256, max_positions=1024,
        convolutions=((512, 3),) * 8, attention=True, dropout=0.1,
        selfattention=False, attention_nheads=1, selfattention_nheads=1,
        project_input=False, gated_attention=False, downsample=False,
        pretrained=False, trained_decoder=None,
    ):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        self.pretrained = pretrained
        self.pretrained_decoder = trained_decoder
        self.dropout = dropout
        self.need_attn = True
        in_channels = convolutions[0][0]

        def expand_bool_array(val):
            if isinstance(val, bool):
                # expand True into [True, True, ...] and do the same with False
                return [val] * len(convolutions)
            return val

        attention = expand_bool_array(attention)
        selfattention = expand_bool_array(selfattention)

        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
        )

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.selfattention = nn.ModuleList()
        self.attproj = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            self.projections.append(
                Linear(in_channels, out_channels) if in_channels != out_channels else None
            )
            self.convolutions.append(
                LinearizedConv1d(
                    in_channels, out_channels * 2, kernel_size,
                    padding=(kernel_size - 1), dropout=dropout,
                )
            )

            self.attention.append(
                DownsampledMultiHeadAttention(
                    out_channels, embed_dim, attention_nheads,
                    project_input=project_input, gated=False, downsample=False,
                ) if attention[i] else None
            )

            self.attproj.append(
                Linear(out_channels, embed_dim, dropout=dropout) if attention[i] else None
            )
            self.selfattention.append(
                SelfAttention(
                    out_channels, embed_dim, selfattention_nheads,
                    project_input=project_input, gated=gated_attention,
                    downsample=downsample,
                ) if selfattention[i] else None
            )
            in_channels = out_channels

        self.fc2 = Linear(in_channels, out_embed_dim)
        self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

        # model fusion
        if self.pretrained:
            # independent gates are learned from the concatenated input
            self.gate1 = nn.Sequential(Linear(out_embed_dim*2, out_embed_dim), nn.Sigmoid())
            self.gate2 = nn.Sequential(Linear(out_embed_dim*2, out_embed_dim), nn.Sigmoid())
            # pretrained and trained models are joined
            self.joining = nn.Sequential(
                Linear(out_embed_dim*2, out_embed_dim*2),
                LayerNorm(out_embed_dim*2),
                nn.GLU(),
                Linear(out_embed_dim, out_embed_dim*2),
                LayerNorm(out_embed_dim*2),
                nn.GLU(),
                Linear(out_embed_dim, out_embed_dim),
                LayerNorm(out_embed_dim)
            )
            # pretrained model contains an output layer that is nhid -> vocab size
            # but the models are combined in their hidden state
            # the hook stores the output of the pretrained model forward
            self.pretrained_outputs = {}

            def save_output():
                def hook(a, b, output):
                    self.pretrained_outputs["out"] = output
                return hook

            self.pretrained_decoder.fc2.register_forward_hook(save_output())

    def forward(self, prev_output_tokens, encoder_out):
        trained_encoder_out = encoder_out['pretrained'] if self.pretrained else None
        encoder_out = encoder_out['encoder']['encoder_out']

        encoder_a, encoder_b = self._split_encoder_out(encoder_out)

        # embed positions
        positions = self.embed_positions(prev_output_tokens)

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens) + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x.transpose(0, 1)

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        avg_attn_scores = None
        for proj, conv, attention, selfattention, attproj in zip(
            self.projections, self.convolutions, self.attention, self.selfattention, self.attproj
        ):
            residual = x if proj is None else proj(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                r = x
                x, attn_scores = attention(attproj(x) + target_embedding, encoder_a, encoder_b)
                x = x + r
                if not self.training and self.need_attn:
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

            if selfattention is not None:
                x = selfattention(x)

            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if not self.pretrained:
            x = self.fc3(x)

        # fusion gating
        if self.pretrained:
            trained_x, _ = self.pretrained_decoder.forward(prev_output_tokens, trained_encoder_out)
            y = torch.cat([x, self.pretrained_outputs["out"]], dim=-1)
            gate1 = self.gate1(y)
            gate2 = self.gate2(y)
            gated_x1 = gate1 * x
            gated_x2 = gate2 * self.pretrained_outputs["out"]
            fusion = torch.cat([gated_x1, gated_x2], dim=-1)
            fusion = self.joining(fusion)
            fusion_output = self.fc3(fusion)
            return fusion_output, avg_attn_scores
        else:
            return x, avg_attn_scores

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _split_encoder_out(self, encoder_out):
        """Split and transpose encoder outputs."""
        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(0, 1).contiguous()
        encoder_b = encoder_b.transpose(0, 1).contiguous()
        result = (encoder_a, encoder_b)
        return result


class SelfAttention(nn.Module):

    def __init__(self, out_channels, embed_dim, num_heads, project_input=False, gated=False, downsample=False):
        super().__init__()
        self.attention = DownsampledMultiHeadAttention(
            out_channels, embed_dim, num_heads, dropout=0, bias=True,
            project_input=project_input, gated=gated, downsample=downsample,
        )
        self.in_proj_q = Linear(out_channels, embed_dim)
        self.in_proj_k = Linear(out_channels, embed_dim)
        self.in_proj_v = Linear(out_channels, embed_dim)
        self.ln = LayerNorm(out_channels)

    def forward(self, x):
        residual = x
        query = self.in_proj_q(x)
        key = self.in_proj_k(x)
        value = self.in_proj_v(x)
        x, _ = self.attention(query, key, value, mask_future_timesteps=True, use_scalar_bias=True)
        return self.ln(x + residual)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return m


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0., **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return m


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return m


@register_model_architecture('fconv_self_att', 'fconv_self_att')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 3')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 8')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.self_attention = getattr(args, 'self_attention', 'False')
    args.encoder_attention = getattr(args, 'encoder_attention', 'False')
    args.multihead_attention_nheads = getattr(args, 'multihead_attention_nheads', 1)
    args.multihead_self_attention_nheads = getattr(args, 'multihead_self_attention_nheads', 1)
    args.encoder_attention_nheads = getattr(args, 'encoder_attention_nheads', 1)
    args.project_input = getattr(args, 'project_input', 'False')
    args.gated_attention = getattr(args, 'gated_attention', 'False')
    args.downsample = getattr(args, 'downsample', 'False')
    args.pretrained_checkpoint = getattr(args, 'pretrained_checkpoint', '')
    args.pretrained = getattr(args, 'pretrained', 'False')


@register_model_architecture('fconv_self_att', 'fconv_self_att_wp')
def fconv_self_att_wp(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(128, 3)] * 2 + [(512,3)] * 1')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 4)] * 4 + [(768, 4)] * 2 + [(1024, 4)] * 1')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.self_attention = getattr(args, 'self_attention', 'True')
    args.multihead_self_attention_nheads = getattr(args, 'multihead_self_attention_nheads', 4)
    args.project_input = getattr(args, 'project_input', 'True')
    args.gated_attention = getattr(args, 'gated_attention', 'True')
    args.downsample = getattr(args, 'downsample', 'True')
    base_architecture(args)
