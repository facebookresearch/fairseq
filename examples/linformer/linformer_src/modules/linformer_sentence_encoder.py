# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn as nn
from fairseq.modules import TransformerSentenceEncoder

from .linformer_sentence_encoder_layer import LinformerSentenceEncoderLayer


class LinformerSentenceEncoder(TransformerSentenceEncoder):
    """
    Implementation for a Bi-directional Linformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    LinformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
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
        layerdrop: float = 0.0,
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
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        compressed: int = 4,
        shared_kv_compressed: int = 0,
        shared_layer_kv_compressed: int = 0,
        freeze_compress: int = 0,
    ) -> None:

        # Initialize linformer parameters
        self.compressed = compressed
        self.shared_kv_compressed = shared_kv_compressed
        self.shared_layer_kv_compressed = shared_layer_kv_compressed
        self.compress_layer = None
        self.freeze_compress = freeze_compress

        super().__init__(
            padding_idx=padding_idx,
            vocab_size=vocab_size,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            layerdrop=layerdrop,
            max_seq_len=max_seq_len,
            num_segments=num_segments,
            use_position_embeddings=use_position_embeddings,
            offset_positions_by_padding=offset_positions_by_padding,
            encoder_normalize_before=encoder_normalize_before,
            apply_bert_init=apply_bert_init,
            activation_fn=activation_fn,
            learned_pos_embedding=learned_pos_embedding,
            embed_scale=embed_scale,
            freeze_embeddings=freeze_embeddings,
            n_trans_layers_to_freeze=n_trans_layers_to_freeze,
            export=export,
            traceable=traceable,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

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
        q_noise,
        qn_block_size,
    ):
        if self.shared_layer_kv_compressed == 1:
            compress_layer = nn.Linear(
                self.max_seq_len, self.max_seq_len // self.compressed
            )
            # intialize parameters for compressed layer
            nn.init.xavier_uniform_(compress_layer.weight, gain=1 / math.sqrt(2))
            if self.freeze_compress == 1:
                compress_layer.weight.requires_grad = False
            self.compress_layer = compress_layer

        return LinformerSentenceEncoderLayer(
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
            compressed=self.compressed,
            max_seq_len=self.max_seq_len,
            shared_kv_compressed=self.shared_kv_compressed,
            shared_compress_layer=(
                None if self.shared_layer_kv_compressed == 0 else self.compress_layer
            ),
            freeze_compress=self.freeze_compress,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []

        # update key name for shared layer in new version of code
        for k in state_dict.keys():
            if k.startswith(prefix + "compress_layer"):
                if self.shared_layer_kv_compressed:
                    for layer_idx in range(len(self.layers)):
                        new_k = prefix + "layers.{0}.shared_compress_layer.{1}".format(
                            layer_idx,
                            k[len(prefix + "compress_layer.") :],
                        )
                        items_to_add[new_k] = state_dict[k]

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
