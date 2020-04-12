# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from fairseq.modules import TransformerSentenceEncoder
from fairseq.modules.sparse_transformer_sentence_encoder_layer import SparseTransformerSentenceEncoderLayer


class SparseTransformerSentenceEncoder(TransformerSentenceEncoder):
    """
    Sparse implementation of the TransformerSentenceEncoder
    - see SparseMultiheadAttention
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        is_bidirectional: bool = True,
        stride: int = 32,
        expressivity: int = 8,
    ) -> None:

        super().__init__(
            padding_idx, vocab_size, num_encoder_layers, embedding_dim,
            ffn_embedding_dim, num_attention_heads, dropout, attention_dropout,
            activation_dropout, max_seq_len, num_segments, use_position_embeddings,
            offset_positions_by_padding, encoder_normalize_before, apply_bert_init,
            activation_fn, learned_pos_embedding, embed_scale, freeze_embeddings,
            n_trans_layers_to_freeze, export
        )

        self.layers = nn.ModuleList(
            [
                SparseTransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    is_bidirectional=is_bidirectional,
                    stride=stride,
                    expressivity=expressivity,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])
