# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.modules import TransformerSentenceEncoderLayer
from fairseq.modules.fb_multihead_linear_attention import MultiheadLinearAttention
from fairseq import utils
from fairseq.modules import LayerNorm

from torch import nn


class LinformerSentenceEncoderLayer(TransformerSentenceEncoderLayer):
    """
    Implements a Linformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        compressed: int = 1,
        max_seq_len: int = 256,
        shared_kv_compressed: int = 0,
        shared_compress_layer: any = None,
        freeze_compress: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize linformer parameters
        self.compressed = compressed
        self.max_seq_len = max_seq_len
        self.shared_kv_compressed = shared_kv_compressed
        self.freeze_compress = freeze_compress

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            shared_compress_layer=shared_compress_layer,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        shared_compress_layer,
    ):
        return MultiheadLinearAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            compressed=self.compressed,
            max_seq_len=self.max_seq_len,
            shared_kv_compressed=self.shared_kv_compressed,
            shared_compress_layer=shared_compress_layer,
            freeze_compress=self.freeze_compress,
        )
