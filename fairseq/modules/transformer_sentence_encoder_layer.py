# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import gelu, MultiheadAttention, BertLayerNorm, LayerNorm


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.

    If the flag use_bert_layer_norm is set then we use the custom
    BertLayerNorm module instead of LayerNorm.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        encoder_normalize_before: bool = False,
        use_bert_layer_norm: bool = False,
        use_gelu: bool = True,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.normalize_before = encoder_normalize_before

        # Initialize blocks
        self.activation_fn = gelu if use_gelu else F.relu
        self.self_attn = MultiheadAttention(
            self.embedding_dim, num_attention_heads, dropout=attention_dropout
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = (
            BertLayerNorm(self.embedding_dim)
            if use_bert_layer_norm
            else LayerNorm(self.embedding_dim, eps=1e-12)
        )
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = (
            BertLayerNorm(self.embedding_dim)
            if use_bert_layer_norm
            else LayerNorm(self.embedding_dim, eps=1e-12)
        )

    def _maybe_layer_norm(
        self,
        layer_norm: nn.Module,
        x: torch.Tensor,
        before: bool = False,
        after: bool = False,
    ):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """

        residual = x
        x = self._maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self._maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn
