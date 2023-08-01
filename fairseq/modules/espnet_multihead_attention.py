#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import torch
from torch import nn

from fairseq.modules.rotary_positional_embedding import (
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
)


class ESPNETMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        dropout: Dropout rate.
    """

    def __init__(self, n_feat, n_head, dropout):
        """Construct an MultiHeadedAttention object."""
        super(ESPNETMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward_qkv(self, query, key, value, **kwargs):
        """Transform query, key and value.
        Args:
            query: Query tensor  B X T1 X C
            key: Key tensor B X T2 X C
            value: Value tensor  B X T2 X C
        Returns:
            torch.Tensor: Transformed query tensor  B X n_head X T1 X d_k
            torch.Tensor: Transformed key tensor B X n_head X T2 X d_k
            torch.Tensor: Transformed value tensor  B X n_head X T2 X d_k
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value: Transformed value B X n_head X T2 X d_k.
            scores: Attention score  B X n_head X T1 X T2
            mask: Mask  T2 X B
        Returns:
            torch.Tensor: Transformed value  B X T1 X d_model
                weighted by the attention score  B X T1 X T2
        """
        n_batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2).to(bool),
                float("-inf"),  # (batch, head, time1, time2)
            )
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor T X B X C
            key (torch.Tensor): Key tensor T X B X C
            value (torch.Tensor): Value tensor T X B X C
            mask (torch.Tensor): Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None


class RelPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        dropout: Dropout rate.
        zero_triu: Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_feat, n_head, dropout, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head, dropout)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x: Input tensor B X n_head X T X 2T-1
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            pos_emb: Positional embedding tensor B X 2T-1 X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X C.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        pos_emb = pos_emb.transpose(0, 1)
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None


class RotaryPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    def __init__(
        self,
        n_feat,
        n_head,
        dropout,
        precision,
        rotary_emd_base=10000,
    ):
        """Construct an RotaryPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head, dropout)
        precision = torch.float
        self.rotary_ndims = self.d_k  # also try self.d_k//2
        if precision == "fp16":
            precision = torch.half

        self.rotary_emb = RotaryPositionalEmbedding(
            self.rotary_ndims, base=rotary_emd_base, precision=precision
        )

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute rotary position attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        Notes:
            Assumes self attn
        """

        T, B, C = value.size()
        query = query.view(T, B, self.h, self.d_k)
        key = key.view(T, B, self.h, self.d_k)
        value = value.view(T, B, self.h, self.d_k)
        cos, sin = self.rotary_emb(value, seq_len=T)
        query, key = apply_rotary_pos_emb(
            query, key, cos, sin, offset=0
        )  # offset is based on layer_past

        query = query.view(T, B, self.h * self.d_k)
        key = key.view(T, B, self.h * self.d_k)
        value = value.view(T, B, self.h * self.d_k)

        # TBD to BTD
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None
