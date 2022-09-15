# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import torch
import torch.nn as nn


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


def infer_conv_output_dim(in_channels, input_dim, out_channels):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
    x = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
    x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
    x = x.transpose(1, 2)
    mb, seq = x.size()[:2]
    return x.contiguous().view(mb, seq, -1).size(-1)


class Conv2dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 2D convolution based on ESPnet implementation
    (https://github.com/espnet/espnet)

    Args:
        input_channels (int): the number of input channels
        input_feat_per_channel (int): encoder input dimension per input channel
        conv_out_channels (int): the number of output channels of conv layer
        encoder_embed_dim (int): encoder dimentions
    """

    def __init__(
        self,
        input_channels: int,
        input_feat_per_channel: int,
        conv_out_channels: int,
        encoder_embed_dim: int,
    ):
        super().__init__()
        assert input_channels == 1, input_channels
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels, conv_out_channels, 3, stride=2, padding=3 // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                conv_out_channels,
                conv_out_channels,
                3,
                stride=2,
                padding=3 // 2,
            ),
            torch.nn.ReLU(),
        )
        transformer_input_dim = infer_conv_output_dim(
            input_channels, input_feat_per_channel, conv_out_channels
        )
        self.out = torch.nn.Linear(transformer_input_dim, encoder_embed_dim)

    def forward(self, src_tokens, src_lengths):
        B, T_i, C = src_tokens.size()
        x = src_tokens.view(B, T_i, 1, C).transpose(1, 2).contiguous()
        x = self.conv(x)
        B, _, T_o, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(T_o, B, -1)
        x = self.out(x)

        subsampling_factor = int(T_i * 1.0 / T_o + 0.5)
        input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
        input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(
            input_len_0.device
        )
        input_lengths = torch.min(input_len_0, input_len_1)
        return x, input_lengths
