# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.model_parallel.modules import ModelParallelMultiheadAttention
from fairseq.modules import TransformerSentenceEncoderLayer


try:
    from fairseq.model_parallel.megatron.mpu import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


class ModelParallelTransformerSentenceEncoderLayer(TransformerSentenceEncoderLayer):
    """
    Implements a Model Parallel Transformer Encoder Layer used in
    BERT/XLM style pre-trained models.
    """

    def build_fc1(self, input_dim, output_dim, **unused):
        return ColumnParallelLinear(input_dim, output_dim, gather_output=False)

    def build_fc2(self, input_dim, output_dim, **unused):
        return RowParallelLinear(input_dim, output_dim, input_is_parallel=True)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        **kwargs,
    ):
        return ModelParallelMultiheadAttention(
            embed_dim, num_attention_heads, dropout=dropout, self_attention=True
        )

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
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        return x, None
