# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils


class BidirectionalMultiheadSelfAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, mask_curr_state=True):
        super().__init__()
        self.onnx_trace = False
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask_curr_state = mask_curr_state
        self.head_dim = embed_dim // num_heads
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, fwd_x, bwd_x, key_padding_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        assert fwd_x.size() == bwd_x.size()

        tgt_len, bsz, embed_dim = fwd_x.size()
        assert embed_dim == self.embed_dim

        padded_fwd_x = torch.cat([fwd_x.new_zeros(1, bsz, embed_dim), fwd_x])
        padded_bwd_x = torch.cat([bwd_x, bwd_x.new_zeros(1, bsz, embed_dim)])

        q = padded_fwd_x[:-1] + padded_bwd_x[1:]
        kv = torch.cat([fwd_x, bwd_x], dim=0)

        src_len = tgt_len * 2

        q = self.in_proj_q(q)
        k, v = self.in_proj_kv(kv)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if self.mask_curr_state:
            attn_weights += self.mask(attn_weights).unsqueeze(0)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.repeat(1, 2).unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.repeat(1, 2).unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)

    def mask(self, tensor):
        _, half_dim, dim = tensor.size()
        if self.onnx_trace:
            # triu and tril are not supported in onnx
            a = torch._dim_arange(tensor, 2).unsqueeze(0).repeat(half_dim, 1)
            b = torch._dim_arange(tensor, 1).unsqueeze(1).repeat(1, dim)
            mask = (a > b + half_dim).float() + (a < b).float()
            mask = torch.where(
                mask > 0,
                torch.Tensor([0]).type_as(tensor),
                torch.Tensor([float("-Inf")]).type_as(tensor)
            )
        else:
            ones = tensor.new_ones(half_dim, dim).byte()
            mask = ones.triu(half_dim + 1) + ones.tril(-1)
            mask = utils.fill_with_neg_inf(tensor.new(mask.size())).masked_fill_(mask, 0)
        return mask
