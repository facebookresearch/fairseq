# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from fairseq import utils
from fairseq.modules import TransformerSentenceEncoderLayer

from .multihead_linear_attention import MultiheadLinearAttention


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
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        compressed: int = 1,
        max_seq_len: int = 256,
        shared_kv_compressed: int = 0,
        shared_compress_layer: any = None,
        freeze_compress: int = 0,
    ) -> None:

        # Initialize linformer parameters
        self.compressed = compressed
        self.max_seq_len = max_seq_len
        self.shared_kv_compressed = shared_kv_compressed
        self.freeze_compress = freeze_compress

        # wrap in a list so it's not automatically registered by PyTorch
        self.shared_compress_layer = [shared_compress_layer]

        super().__init__(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.register_buffer("version", torch.tensor(2))

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
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
            shared_compress_layer=self.shared_compress_layer[0],
            freeze_compress=self.freeze_compress,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # some old checkpoints had weight sharing implemented incorrectly
        # (note: this was correct in the original paper code)
        if utils.item(state_dict.get(f"{prefix}version", torch.tensor(1))) < 2:
            state_dict[f"{prefix}version"] = torch.tensor(1)
            # check compression layer sharing
            if f"{prefix}shared_compress_layer.weight" in state_dict:
                # reinitialize block without sharing compression layer to match
                # old behavior
                self.shared_compress_layer = [
                    torch.nn.Linear(
                        self.shared_compress_layer[0].weight.size(1),
                        self.shared_compress_layer[0].weight.size(0),
                    )
                ]
                self.self_attn = self.build_self_attention(
                    self.embedding_dim,
                    self.num_attention_heads,
                    dropout=self.attention_dropout,
                    self_attention=True,
                    q_noise=self.q_noise,
                    qn_block_size=self.qn_block_size,
                )
                # delete shared_compress_layer, since it's already copied to
                # self_attn.compress_k.weight
                del state_dict[f"{prefix}shared_compress_layer.weight"]
                if f"{prefix}shared_compress_layer.bias" in state_dict:
                    del state_dict[f"{prefix}shared_compress_layer.bias"]
