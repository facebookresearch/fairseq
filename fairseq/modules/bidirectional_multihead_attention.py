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

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, concat_final_q=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.concat_final_q = concat_final_q

        chunks = 3
        if concat_final_q:
            self.q_proj=nn.Linear(embed_dim*2, embed_dim, bias=bias)
            chunks = 2
        else:
            self.q_proj = None

        self.in_proj_weight = Parameter(torch.Tensor(chunks * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(chunks * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.q_proj is not None:
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            if self.q_proj is not None:
                nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, fwd_x, bwd_x, mask_curr_state=True, key_padding_mask=None):
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

        fwd_idxs = torch.arange(tgt_len, out=fwd_x.new().long())
        bwd_idxs = torch.arange(1, tgt_len + 1, out=fwd_x.new().long())

        if self.concat_final_q:
            q = torch.cat([padded_fwd_x[fwd_idxs], padded_bwd_x[bwd_idxs]], dim=-1)
        else:
            q = padded_fwd_x[fwd_idxs] + padded_bwd_x[bwd_idxs]
        kv = torch.cat([fwd_x, bwd_x], dim=0)

        src_len = tgt_len * 2

        q = self.in_proj_q(q)
        k, v = self.in_proj_kv(kv)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if mask_curr_state:
            attn_weights += self.mask(attn_weights, mask_curr_state).unsqueeze(0)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
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
        if self.concat_final_q:
            return self.q_proj(query)
        else:
            return self._in_proj(query, end=self.embed_dim)

    def in_proj_kv(self, key):
        start = None if self.q_proj is not None else self.embed_dim
        return self._in_proj(key, start=start).chunk(2, dim=-1)

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

    def mask(self, tensor, mask_curr):
        dim = tensor.size(-1)
        half_dim = dim // 2

        add = 1 if mask_curr else 0

        ones = tensor.new_ones(half_dim, dim).byte()
        mask = ones.triu(half_dim + add) + ones.tril(-add)
        mask = utils.fill_with_neg_inf(tensor.new(mask.size())).masked_fill_(mask, 0)
        return mask
