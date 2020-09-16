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
import math

from fairseq import utils


def gaussian(matrix, batch_size=1, variance=1e-6):
    num = (matrix * matrix).unsqueeze(0)
    denom = (2 * variance * variance).unsqueeze(1).unsqueeze(2).repeat(batch_size, 1, 1)
    return  num / denom


def log(matrix, batch_size=1, unused=None):
    return torch.max(torch.zeros_like(matrix), torch.log(matrix))


class LocalMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    Penalties = {'log': log,
                 'gauss': gaussian}

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, penalty='log', **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self._mask = None

        self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.penalty = type(self).Penalties[penalty]

        if penalty == 'gauss':
            self.vars = Parameter(torch.Tensor(self.num_heads).fill_(kwargs['init_variance']))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None, attn_mask=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        if embed_dim != self.embed_dim:
            print("| x: {}, multi_head: {}".format(embed_dim, self.embed_dim))
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if saved_state is not None:
            if 'prev_key' in saved_state:
                k = torch.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = torch.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0).clamp(-1e8)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).bool(),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        # Local attention
        pos_diff = torch.abs(torch.arange(tgt_len).unsqueeze(1) - torch.arange(src_len).unsqueeze(0)).type_as(attn_weights)
        variance = self.vars if hasattr(self, 'vars') else None
        local_mask = self.penalty(pos_diff, bsz, variance)
        attn_weights = attn_weights - local_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2*self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2*self.embed_dim)

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

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(utils.fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )


class ConvAttention2D(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, batch_norm=False, unidirectional=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.unidirectional = unidirectional
        # TODO: Set padding=1 now to avoid short utterence error
        self.padding = 1
        self.head_dim = embed_dim  # // num_heads
        # assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self._mask = None

        self.in_proj_weight = Parameter(torch.Tensor(3 * num_heads, embed_dim, 3, 3))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * num_heads))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Conv2d(
            2 * self.num_heads,
            embed_dim,
            3,
            padding=self.padding,
            bias=bias,
        )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_q = BatchNorm(self.num_heads)
            self.bn_k = BatchNorm(self.num_heads)
            self.bn_v = BatchNorm(self.num_heads)
            self.bn_out = BatchNorm(embed_dim)
        self.relu = F.relu

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        bsz, channels, tgt_len, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        tgt_len = q.size(2)
        src_len = k.size(2)
        freq_len = k.size(3)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.batch_norm:
            q = self.relu(self.bn_q(q.contiguous()).view(bsz * self.num_heads, tgt_len, freq_len))
            k = self.relu(self.bn_k(k.contiguous()).view(bsz * self.num_heads, src_len, freq_len))
            v = self.relu(self.bn_v(v.contiguous()).view(bsz * self.num_heads, src_len, freq_len))
        else:
            q = self.relu(q.contiguous()).view(bsz * self.num_heads, tgt_len, freq_len)
            k = self.relu(k.contiguous()).view(bsz * self.num_heads, src_len, freq_len)
            v = self.relu(v.contiguous()).view(bsz * self.num_heads, src_len, freq_len)

        attn_weights_t = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights_t.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if self.unidirectional or (mask_future_timesteps and incremental_state is None):
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights_t += self.buffered_mask(attn_weights_t).unsqueeze(0).clamp(min=-1e8)

        # only apply masking at training time (when incremental state is None)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights_t = attn_weights_t.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights_t = attn_weights_t.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).bool(),
                float('-inf'),
            ).type_as(attn_weights_t)  # FP16 support: cast to float and back
            attn_weights_t = attn_weights_t.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights_t = F.softmax(attn_weights_t.float(), dim=-1).type_as(attn_weights_t)
        attn_weights_t = F.dropout(attn_weights_t, p=self.dropout, training=self.training)

        attn_t = torch.bmm(attn_weights_t, v)
        assert list(attn_t.size()) == [bsz * self.num_heads, tgt_len, freq_len]

        if self.unidirectional:
            assert src_len == tgt_len, "self attention only"
            # q_f, v_s size (bsz * self.num_heads * tgt_len, freq_len, 1)
            q_f = q.view(-1, freq_len, 1)
            # k_f size (bsz * self.num_heads * src_len, 1, freq_len)
            k_f = k.view(-1, 1, freq_len)
            # v_f size (bsz * self.num_heads * src_len, freq_len. 1)
            v_f = k.view(-1, freq_len, 1)
            # attn_weights_f (bsz * self.num_heads * src_len, freq_len, freq_len)
            attn_weights_f = torch.bmm(q_f, k_f)
            # attn_weights_f (bsz * self.num_heads * src_len, freq_len, freq_len)
            attn_weights_f = F.softmax(attn_weights_f.float() + 1e-8, dim=-1).type_as(attn_weights_f)
            attn_weights_f = F.dropout(attn_weights_f, p=self.dropout, training=self.training)
            # attn_f (bsz * self.num_heads * src_len, freq_len)
            attn_f = torch.bmm(attn_weights_f, v_f).unsqueeze(-1)
            # attn_f (bsz * self.num_heads * src_len, freq_len, 1)
            attn_f = attn_f.view(-1, tgt_len, freq_len).transpose(1, 2)
        else:
            q_f = q.transpose(2, 1)
            v_f = v.transpose(2, 1)
            attn_weights_f = torch.bmm(q_f, k)
            assert list(attn_weights_f.size()) == [bsz * self.num_heads, freq_len, freq_len]

            # only apply masking at training time (when incremental state is None)
            attn_weights_f = F.softmax(attn_weights_f.float(), dim=-1).type_as(attn_weights_f)
            attn_weights_f = F.dropout(attn_weights_f, p=self.dropout, training=self.training)

            attn_f = torch.bmm(attn_weights_f, v_f)

        assert list(attn_f.size()) == [bsz * self.num_heads, freq_len, tgt_len]
        attn_t = attn_t.view(bsz, self.num_heads, tgt_len, freq_len).contiguous()
        attn_f = attn_f.transpose(1, 2).view(bsz, self.num_heads, tgt_len, freq_len).contiguous()
        attn = torch.cat([attn_t, attn_f], dim=1).contiguous()
        if self.batch_norm:
            attn = self.relu(self.bn_out(self.out_proj(attn)))
        else:
            attn = self.relu(self.out_proj(attn))

        if need_weights:
            # average attention weights over heads
            attn_weights_t = attn_weights_t.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights_t = attn_weights_t.sum(dim=1) / self.num_heads
        else:
            attn_weights_t = None

        return attn, attn_weights_t

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=1).chunk(2, dim=1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=1)

    def in_proj_k(self, key):
        return self._in_proj(key, start=1, end=2)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :, :, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :, :, :]
            if bias is not None:
                bias = bias[start:]
        return F.conv2d(
            input,
            weight,
            bias=bias,
            padding=self.padding,
        )

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(utils.fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m


def BatchNorm(embedding_dim):
    m = nn.BatchNorm2d(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m
