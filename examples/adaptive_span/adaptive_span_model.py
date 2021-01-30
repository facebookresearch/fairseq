# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.layer_norm import LayerNorm

from .adaptive_span_attention import AdaptiveSpan

# Size notations:
# B = batch_size, H = d_model, M = block_size, L = attn_span


def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X


class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """

    def __init__(self, d_model, n_head, attn_span, dropout, adapt_span_layer, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model  # size of a single head
        self.attn_span = attn_span
        self.adaptive_span = AdaptiveSpan(
            attn_span=attn_span,
            n_head=n_head,
            adapt_span_layer=adapt_span_layer,
            **kargs
        )

    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        key, value, key_pe = self.adaptive_span.trim_memory(query, key, value, key_pe)

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(self.d_model)  # B x M X L_pos

        attn = F.softmax(attn.float(), dim=-1).type_as(attn)

        # trim attention lengths according to the learned span
        attn = self.adaptive_span(attn)

        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H
        return out

    def get_cache_size(self):
        return self.adaptive_span.get_cache_size()


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, d_model, n_head, **kargs):
        nn.Module.__init__(self)
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.attn = SeqAttention(d_model=self.head_dim, n_head=n_head, **kargs)
        self.proj_query = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.proj_query.weight)
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.proj_out.weight)
        self.proj_val = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.proj_val.weight)
        self.proj_key = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.proj_key.weight)

    def head_reshape(self, x):
        K = self.n_head
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.n_head
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_inner, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):
    def __init__(self, d_model, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(d_model=d_model, **kargs)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForwardLayer(d_model=d_model, **kargs)
        self.norm2 = LayerNorm(d_model)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        if self.ff is not None:
            ff_out = self.ff(h)
            out = self.norm2(h + ff_out)  # B x M x H
        else:
            out = h
        return out

    def get_cache_size(self):
        return self.attn.attn.get_cache_size()


class TransformerSeq(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_head,
        n_layer,
        attn_span,
        emb_dropout,
        aux_loss_scaler,
        adapt_span_layer,
        **kargs
    ):
        nn.Module.__init__(self)
        # token embeddings
        self.in_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.in_emb.weight, mean=0, std=d_model ** -0.5)
        self.out_emb = nn.Linear(d_model, vocab_size)
        self.aux_loss_scaler = aux_loss_scaler
        if emb_dropout > 0:
            self.emb_dropout = nn.Dropout(emb_dropout)
        else:
            self.emb_dropout = None
        # position embeddings
        self.key_pe = nn.Parameter(torch.randn(1, d_model // n_head, attn_span))

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                d_model=d_model,
                n_head=n_head,
                attn_span=attn_span,
                adapt_span_layer=adapt_span_layer,
                **kargs
            )
            for _ in range(n_layer)
        )

    def forward(self, x, h_cache, target=None):
        # x size = B x M
        block_size = x.size(1)
        h = self.in_emb(x)  # B x M x H
        if self.emb_dropout is not None:
            h = self.emb_dropout(h)

        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.get_cache_size()
            if cache_size > block_size:
                h_cache_next_l = torch.cat(
                    [h_cache[l][:, -cache_size + block_size :, :], h], dim=1
                ).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], self.key_pe)  # B x M x H

        if self.emb_dropout is not None:
            h = self.emb_dropout(h)

        out = F.log_softmax(self.out_emb(h).float(), dim=-1).type_as(h)
        dummy_loss = None

        return out, h_cache_next, dummy_loss

    def get_aux_loss(self):
        loss = 0.0
        for layer in self.layers:
            loss += layer.attn.attn.adaptive_span.get_loss()
        return self.aux_loss_scaler * loss

    def get_current_max_span(self):
        max_span = 0.0
        for layer in self.layers:
            max_span = max(
                max_span, layer.attn.attn.adaptive_span.get_current_max_span()
            )
        return max_span

    def get_current_avg_span(self):
        avg_span = 0.0
        for layer in self.layers:
            avg_span += layer.attn.attn.adaptive_span.get_current_avg_span()
        return avg_span / len(self.layers)
