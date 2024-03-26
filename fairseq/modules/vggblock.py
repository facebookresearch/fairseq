# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

from collections.abc import Iterable
from itertools import repeat

import torch
import torch.nn as nn


def _pair(v):
    if isinstance(v, Iterable):
        assert len(v) == 2, "len(v) != 2"
        return v
    return tuple(repeat(v, 2))


def infer_conv_output_dim(conv_op, input_dim, sample_inchannel):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, sample_inchannel, sample_seq_len, input_dim)
    # N x C x H x W
    # N: sample_bsz, C: sample_inchannel, H: sample_seq_len, W: input_dim
    x = conv_op(x)
    # N x C x H x W
    x = x.transpose(1, 2)
    # N x H x C x W
    bsz, seq = x.size()[:2]
    per_channel_dim = x.size()[3]
    # bsz: N, seq: H, CxW the rest
    return x.contiguous().view(bsz, seq, -1).size(-1), per_channel_dim


class VGGBlock(torch.nn.Module):
    """
    VGG motibated cnn module https://arxiv.org/pdf/1409.1556.pdf

    Args:
        in_channels: (int) number of input channels (typically 1)
        out_channels: (int) number of output channels
        conv_kernel_size: convolution channels
        pooling_kernel_size: the size of the pooling window to take a max over
        num_conv_layers: (int) number of convolution layers
        input_dim: (int) input dimension
        conv_stride: the stride of the convolving kernel.
            Can be a single number or a tuple (sH, sW)  Default: 1
        padding: implicit paddings on both sides of the input.
            Can be a single number or a tuple (padH, padW). Default: None
        layer_norm: (bool) if layer norm is going to be applied. Default: False

    Shape:
        Input: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
        Output: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size,
        pooling_kernel_size,
        num_conv_layers,
        input_dim,
        conv_stride=1,
        padding=None,
        layer_norm=False,
    ):
        assert (
            input_dim is not None
        ), "Need input_dim for LayerNorm and infer_conv_output_dim"
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = _pair(conv_kernel_size)
        self.pooling_kernel_size = _pair(pooling_kernel_size)
        self.num_conv_layers = num_conv_layers
        self.padding = (
            tuple(e // 2 for e in self.conv_kernel_size)
            if padding is None
            else _pair(padding)
        )
        self.conv_stride = _pair(conv_stride)

        self.layers = nn.ModuleList()
        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(
                in_channels if layer == 0 else out_channels,
                out_channels,
                self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.padding,
            )
            self.layers.append(conv_op)
            if layer_norm:
                conv_output_dim, per_channel_dim = infer_conv_output_dim(
                    conv_op, input_dim, in_channels if layer == 0 else out_channels
                )
                self.layers.append(nn.LayerNorm(per_channel_dim))
                input_dim = per_channel_dim
            self.layers.append(nn.ReLU())

        if self.pooling_kernel_size is not None:
            pool_op = nn.MaxPool2d(kernel_size=self.pooling_kernel_size, ceil_mode=True)
            self.layers.append(pool_op)
            self.total_output_dim, self.output_dim = infer_conv_output_dim(
                pool_op, input_dim, out_channels
            )

    def forward(self, x):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x)
        return x
