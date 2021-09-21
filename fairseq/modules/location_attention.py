# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F


class LocationAttention(nn.Module):
    """
    Attention-Based Models for Speech Recognition
    https://arxiv.org/pdf/1506.07503.pdf

    :param int encoder_dim: # projection-units of encoder
    :param int decoder_dim: # units of decoder
    :param int attn_dim: attention dimension
    :param int conv_dim: # channels of attention convolution
    :param int conv_kernel_size: filter size of attention convolution
    """

    def __init__(self, attn_dim, encoder_dim, decoder_dim,
                 attn_state_kernel_size, conv_dim, conv_kernel_size,
                 scaling=2.0):
        super(LocationAttention, self).__init__()
        self.attn_dim = attn_dim
        self.decoder_dim = decoder_dim
        self.scaling = scaling
        self.proj_enc = nn.Linear(encoder_dim, attn_dim)
        self.proj_dec = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.proj_attn = nn.Linear(conv_dim, attn_dim, bias=False)
        self.conv = nn.Conv1d(attn_state_kernel_size, conv_dim,
                              2 * conv_kernel_size + 1,
                              padding=conv_kernel_size, bias=False)
        self.proj_out = nn.Sequential(nn.Tanh(), nn.Linear(attn_dim, 1))

        self.proj_enc_out = None  # cache

    def clear_cache(self):
        self.proj_enc_out = None

    def forward(self, encoder_out, encoder_padding_mask, decoder_h, attn_state):
        """
        :param torch.Tensor encoder_out: padded encoder hidden state B x T x D
        :param torch.Tensor encoder_padding_mask: encoder padding mask
        :param torch.Tensor decoder_h: decoder hidden state B x D
        :param torch.Tensor attn_prev: previous attention weight B x K x T
        :return: attention weighted encoder state (B, D)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T)
        :rtype: torch.Tensor
        """
        bsz, seq_len, _ = encoder_out.size()
        if self.proj_enc_out is None:
            self.proj_enc_out = self.proj_enc(encoder_out)

        # B x K x T -> B x C x T
        attn = self.conv(attn_state)
        # B x C x T -> B x T x C -> B x T x D
        attn = self.proj_attn(attn.transpose(1, 2))

        if decoder_h is None:
            decoder_h = encoder_out.new_zeros(bsz, self.decoder_dim)
        dec_h = self.proj_dec(decoder_h).view(bsz, 1, self.attn_dim)

        out = self.proj_out(attn + self.proj_enc_out + dec_h).squeeze(2)
        out.masked_fill_(encoder_padding_mask, -float("inf"))

        w = F.softmax(self.scaling * out, dim=1)
        c = torch.sum(encoder_out * w.view(bsz, seq_len, 1), dim=1)
        return c, w
