# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import BeamableMM


class FConvModel(nn.Module):
    def __init__(self, encoder, decoder, padding_idx=1):
        super(FConvModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.num_attention_layers = sum([layer is not None for layer in decoder.attention])
        self.padding_idx = padding_idx
        self._is_generation_fast = False

    def forward(self, src_tokens, src_positions, input_tokens, input_positions):
        encoder_out = self.encoder(src_tokens, src_positions)
        decoder_out, _ = self.decoder(input_tokens, input_positions, encoder_out)
        return decoder_out.view(-1, decoder_out.size(-1))

    def make_generation_fast_(self, use_beamable_mm=False):
        """Optimize model for faster generation.

        Optimizations include:
        - remove WeightNorm
        - (optionally) use BeamableMM in attention layers

        The optimized model should not be used again for training.

        Note: this can be combined with incremental inference in the Decoder for
        even faster generation.
        """
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)

        # use BeamableMM in attention layers
        if use_beamable_mm:
            self.decoder._use_beamable_mm()

        def train(mode):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train


class Encoder(nn.Module):
    """Convolutional encoder"""
    def __init__(self, num_embeddings, embed_dim=512, max_positions=1024,
                 convolutions=((512, 3),) * 20, dropout=0.1, padding_idx=1):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = None
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = Embedding(max_positions, embed_dim, padding_idx)

        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        for (out_channels, kernel_size) in convolutions:
            pad = (kernel_size - 1) // 2
            self.projections.append(Projection(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                Conv1d(in_channels, out_channels * 2, kernel_size, padding=pad,
                       dropout=dropout))
            in_channels = out_channels
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, tokens, positions):
        # embed tokens and positions
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=1)
            x = (x + residual) * math.sqrt(0.5)

        # B x T x C -> B x T x C
        x = x.transpose(2, 1)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        x = grad_multiply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return x, y


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super(AttentionLayer, self).__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores


class Decoder(nn.Module):
    """Convolutional decoder"""
    def __init__(self, num_embeddings, embed_dim=512, out_embed_dim=256,
                 max_positions=1024, convolutions=((512, 3),) * 20,
                 attention=True, dropout=0.1, padding_idx=1):
        super(Decoder, self).__init__()
        self.dropout = dropout

        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)

        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = Embedding(max_positions, embed_dim, padding_idx)
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            pad = kernel_size - 1
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                Conv1d(in_channels, out_channels * 2, kernel_size,
                       padding=pad, dropout=dropout))
            self.attention.append(AttentionLayer(out_channels, embed_dim)
                                  if attention[i] else None)
            in_channels = out_channels
        self.fc2 = Linear(in_channels, out_embed_dim)
        self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

        self._is_inference_incremental = False

    def forward(self, tokens, positions, encoder_out):
        # embed tokens and positions
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            residual = x if proj is None else proj(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x.transpose(1, 2)).transpose(2, 1)
            if conv.padding[0] > 0:
                x = x[:, :-conv.padding[0], :]  # remove future timestamps
            x = F.glu(x)

            # attention
            if attention is not None:
                x, attn_scores = attention(x, target_embedding, (encoder_a, encoder_b))
                attn_scores = attn_scores / num_attn_layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores += attn_scores

            # residual
            x = (x + residual) * math.sqrt(0.5)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x, avg_attn_scores

    def context_size(self):
        """Maximum number of input elements each output element depends on"""
        context = 1
        for conv in self.convolutions:
            context += conv.kernel_size[0] - 1
        return context

    def _use_beamable_mm(self):
        """Replace torch.bmm with BeamableMM in attention layers."""
        beamable_mm = BeamableMM()
        for attn in self.attention:
            attn.bmm = beamable_mm


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def Projection(in_features, out_features, dropout=0):
    """Weight-normalized Linear via 1x1 convolution (input: N x C x T)"""
    m = nn.Conv1d(in_features, out_features, kernel_size=1)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def grad_multiply(x, scale):
    return GradMultiply.apply(x, scale)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def get_archs():
    return [
        'fconv', 'fconv_iwslt_de_en', 'fconv_wmt_en_ro', 'fconv_wmt_en_de', 'fconv_wmt_en_fr',
    ]


def _check_arch(args):
    """Check that the specified architecture is valid and not ambiguous."""
    if args.arch not in get_archs():
        raise ValueError('Unknown fconv model architecture: {}'.format(args.arch))
    if args.arch != 'fconv':
        # check that architecture is not ambiguous
        for a in ['encoder_embed_dim', 'encoder_layers', 'decoder_embed_dim', 'decoder_layers',
                  'decoder_out_embed_dim']:
            if hasattr(args, a):
                raise ValueError('--{} cannot be combined with --arch={}'.format(a, args.arch))


def parse_arch(args):
    _check_arch(args)

    if args.arch == 'fconv_iwslt_de_en':
        args.encoder_embed_dim = 256
        args.encoder_layers = '[(256, 3)] * 4'
        args.decoder_embed_dim = 256
        args.decoder_layers = '[(256, 3)] * 3'
        args.decoder_out_embed_dim = 256
    elif args.arch == 'fconv_wmt_en_ro':
        args.encoder_embed_dim = 512
        args.encoder_layers = '[(512, 3)] * 20'
        args.decoder_embed_dim = 512
        args.decoder_layers = '[(512, 3)] * 20'
        args.decoder_out_embed_dim = 512
    elif args.arch == 'fconv_wmt_en_de':
        convs = '[(512, 3)] * 9'       # first 9 layers have 512 units
        convs += ' + [(1024, 3)] * 4'  # next 4 layers have 1024 units
        convs += ' + [(2048, 1)] * 2'  # final 2 layers use 1x1 convolutions
        args.encoder_embed_dim = 768
        args.encoder_layers = convs
        args.decoder_embed_dim = 768
        args.decoder_layers = convs
        args.decoder_out_embed_dim = 512
    elif args.arch == 'fconv_wmt_en_fr':
        convs = '[(512, 3)] * 6'       # first 6 layers have 512 units
        convs += ' + [(768, 3)] * 4'   # next 4 layers have 768 units
        convs += ' + [(1024, 3)] * 3'  # next 3 layers have 1024 units
        convs += ' + [(2048, 1)] * 1'  # next 1 layer uses 1x1 convolutions
        convs += ' + [(4096, 1)] * 1'  # final 1 layer uses 1x1 convolutions
        args.encoder_embed_dim = 768
        args.encoder_layers = convs
        args.decoder_embed_dim = 768
        args.decoder_layers = convs
        args.decoder_out_embed_dim = 512
    else:
        assert args.arch == 'fconv'

    # default architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    return args


def build_model(args, dataset):
    padding_idx = dataset.dst_dict.pad()
    encoder = Encoder(
        len(dataset.src_dict),
        embed_dim=args.encoder_embed_dim,
        convolutions=eval(args.encoder_layers),
        dropout=args.dropout,
        padding_idx=padding_idx,
        max_positions=args.max_positions,
    )
    decoder = Decoder(
        len(dataset.dst_dict),
        embed_dim=args.decoder_embed_dim,
        convolutions=eval(args.decoder_layers),
        out_embed_dim=args.decoder_out_embed_dim,
        attention=eval(args.decoder_attention),
        dropout=args.dropout,
        padding_idx=padding_idx,
        max_positions=args.max_positions,
    )
    return FConvModel(encoder, decoder, padding_idx)
