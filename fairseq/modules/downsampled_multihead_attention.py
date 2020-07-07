# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.scalar_bias import scalar_bias
from fairseq.modules.fairseq_dropout import FairseqDropout


class SingleHeadAttention(nn.Module):
    """
    Single-head attention that supports Gating and Downsampling
    """
    def __init__(
        self, out_channels, embed_dim, head_dim, head_index, dropout=0.,
        bias=True, project_input=True, gated=False, downsample=False,
        num_heads=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.head_index = head_index
        self.head_dim = head_dim
        self.project_input = project_input
        self.gated = gated
        self.downsample = downsample
        self.num_heads = num_heads
        self.projection = None

        k_layers = []
        v_layers = []
        if self.downsample:
            k_layers.append(Downsample(self.head_index))
            v_layers.append(Downsample(self.head_index))
            out_proj_size = self.head_dim
        else:
            out_proj_size = self.head_dim * self.num_heads
        if self.gated:
            k_layers.append(GatedLinear(self.embed_dim, out_proj_size, bias=bias))
            self.in_proj_q = GatedLinear(self.embed_dim, out_proj_size, bias=bias)
            v_layers.append(GatedLinear(self.embed_dim, out_proj_size, bias=bias))
        else:
            k_layers.append(Linear(self.embed_dim, out_proj_size, bias=bias))
            self.in_proj_q = Linear(self.embed_dim, out_proj_size, bias=bias)
            v_layers.append(Linear(self.embed_dim, out_proj_size, bias=bias))

        self.in_proj_k = nn.Sequential(*k_layers)
        self.in_proj_v = nn.Sequential(*v_layers)

        if self.downsample:
            self.out_proj = Linear(out_proj_size, self.head_dim, bias=bias)
        else:
            self.out_proj = Linear(out_proj_size, out_channels, bias=bias)

        self.scaling = self.head_dim**-0.5

    def forward(
        self, query, key, value, mask_future_timesteps=False,
        key_padding_mask=None, use_scalar_bias=False,
    ):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        src_len, bsz, out_channels = key.size()
        tgt_len = query.size(0)
        assert list(query.size()) == [tgt_len, bsz, out_channels]
        assert key.size() == value.size()

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.downsample:
            size = bsz
        else:
            size = bsz * self.num_heads

        k = key
        v = value
        q = query
        if self.project_input:
            q = self.in_proj_q(q)
            k = self.in_proj_k(k)
            v = self.in_proj_v(v)
            src_len = k.size()[0]
        q *= self.scaling

        if not self.downsample:
            q = q.view(tgt_len, size, self.head_dim)
            k = k.view(src_len, size, self.head_dim)
            v = v.view(src_len, size, self.head_dim)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if mask_future_timesteps:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights *= torch.tril(
                attn_weights.data.new([1]).expand(tgt_len, tgt_len).clone(),
                diagonal=-1,
            )[:, ::self.head_index + 1 if self.downsample else 1].unsqueeze(0)
            attn_weights += torch.triu(
                attn_weights.data.new([-math.inf]).expand(tgt_len, tgt_len).clone(),
                diagonal=0
            )[:, ::self.head_index + 1 if self.downsample else 1].unsqueeze(0)
        tgt_size = tgt_len
        if use_scalar_bias:
            attn_weights = scalar_bias(attn_weights, 2)
            v = scalar_bias(v, 1)
            tgt_size += 1

        if key_padding_mask is not None:
            # don't attend to padding symbols
            if key_padding_mask.max() > 0:
                if self.downsample:
                    attn_weights = attn_weights.view(bsz, 1, tgt_len, src_len)
                else:
                    attn_weights = attn_weights.view(size, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -math.inf,
                )
                attn_weights = attn_weights.view(size, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_weights, v)
        if self.downsample:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.head_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        attn = self.out_proj(attn)

        return attn, attn_weights


class DownsampledMultiHeadAttention(nn.ModuleList):
    """
    Multi-headed attention with Gating and Downsampling
    """
    def __init__(
        self, out_channels, embed_dim, num_heads, dropout=0., bias=True,
        project_input=True, gated=False, downsample=False,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.downsample = downsample
        self.gated = gated
        self.project_input = project_input
        assert self.head_dim * num_heads == embed_dim

        if self.downsample:
            attention_heads = []
            for index in range(self.num_heads):
                attention_heads.append(
                    SingleHeadAttention(
                        out_channels, self.embed_dim, self.head_dim, index,
                        dropout, bias, self.project_input, self.gated,
                        self.downsample, self.num_heads,
                    )
                )
            super().__init__(modules=attention_heads)
            self.out_proj = Linear(embed_dim, out_channels, bias=bias)
        else:
            # either we have a list of attention heads, or just one attention head
            # if not being downsampled, we can do the heads with one linear layer instead of separate ones
            super().__init__()
            self.attention_module = SingleHeadAttention(
                out_channels, self.embed_dim, self.head_dim, 1, dropout,
                bias, self.project_input, self.gated, self.downsample, self.num_heads,
            )

    def forward(
        self, query, key, value, mask_future_timesteps=False,
        key_padding_mask=None, use_scalar_bias=False,
    ):
        src_len, bsz, embed_dim = key.size()
        tgt_len = query.size(0)
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        tgt_size = tgt_len
        if use_scalar_bias:
            tgt_size += 1

        attn = []
        attn_weights = []
        if self.downsample:
            for attention_head_number in range(self.num_heads):
                # call the forward of each attention head
                _attn, _attn_weight = self[attention_head_number](
                    query, key, value, mask_future_timesteps, key_padding_mask, use_scalar_bias,
                )
                attn.append(_attn)
                attn_weights.append(_attn_weight)
            full_attn = torch.cat(attn, dim=2)
            full_attn = self.out_proj(full_attn)
            return full_attn, attn_weights[0].clone()
        else:
            _attn, _attn_weight = self.attention_module(
                query, key, value, mask_future_timesteps, key_padding_mask, use_scalar_bias,
            )
            attn.append(_attn)
            attn_weights.append(_attn_weight)
            full_attn = torch.cat(attn, dim=2)
            full_attn_weights = torch.cat(attn_weights)
            full_attn_weights = full_attn_weights.view(bsz, self.num_heads, tgt_size, src_len)
            full_attn_weights = full_attn_weights.sum(dim=1) / self.num_heads
            return full_attn, full_attn_weights


class Downsample(nn.Module):
    """
    Selects every nth element, where n is the index
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[::self.index+1]


def Linear(in_features, out_features, dropout=0., bias=True):
    """Weight-normalized Linear layer (input: B x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def GatedLinear(in_features, out_features, dropout=0., bias=True):
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units"""
    return nn.Sequential(
        Linear(in_features, out_features*4, dropout, bias),
        nn.GLU(),
        Linear(out_features*2, out_features*2, dropout, bias),
        nn.GLU(),
        Linear(out_features, out_features, dropout, bias)
    )
