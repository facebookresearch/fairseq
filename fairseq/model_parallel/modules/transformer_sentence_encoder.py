# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoder,
)

from fairseq.model_parallel.modules import (
    ModelParallelTransformerSentenceEncoderLayer,
)

try:
    from fairseq.model_parallel.megatron.mpu import VocabParallelEmbedding
    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False

import random


class ModelParallelTransformerSentenceEncoder(TransformerSentenceEncoder):
    """
    Implementation for a Model Parallel Bi-directional Transformer based
    Sentence Encoder used in BERT/XLM style pre-trained models.
    """
    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return VocabParallelEmbedding(vocab_size, embedding_dim, padding_idx)

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        **unused,
    ):
        return ModelParallelTransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
        )
