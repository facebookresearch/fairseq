# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)


class TransformerSentenceEncoderTaskemb(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens
        - task_ids: B x 1 matrix representing task ids

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape B x T x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        task_emb_size: int = 768,
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
        activation_fn: str = 'relu',
        learned_pos_embedding: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        embed_scale: float = None,
        export: bool = False,
        task_emb_cond_type: str = 'token'
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.task_emb_cond_type = task_emb_cond_type
        self.task_emb_size = task_emb_size

        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx,
        )
        self.embed_scale = embed_scale

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(
                    self.padding_idx if offset_positions_by_padding
                    else None
                ),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    export=export,
                    task_embedding_dim=task_emb_size if task_emb_cond_type == 'attention' else None
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.task_emb_size != self.embedding_dim:
            self.task_emb_project = nn.Linear(self.task_emb_size, self.embedding_dim)

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

    def forward(
        self,
        tokens: torch.Tensor = None,
        segment_labels: torch.Tensor = None,
        task_embedding: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        self_attn_mask: torch.Tensor = None,
        cls_mask: torch.Tensor = None,
        input_embeddings: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.embed_tokens(tokens)

        if cls_mask is not None:
            cls_mask = cls_mask.view(1, -1, 1)
            cls_embeddings = x * cls_mask

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        else:
            print(padding_mask)
            print('Padding exists')

        if task_embedding is not None:
            if len(list(task_embedding.shape)) == 1:
                task_embedding = task_embedding.unsqueeze(0)

            if self.task_emb_cond_type == 'token':
                if task_embedding.shape[0] == 1:
                    bs = x.shape[0]
                    task_embedding = task_embedding.expand(bs, -1)
                if self.task_emb_size != self.embedding_dim:
                    task_embedding = self.task_emb_project(task_embedding)
                x = torch.cat((x, task_embedding.unsqueeze(1)), dim=1)
                # This is done so that the padding mask is of the right size
                tokens = torch.cat((tokens, tokens[:, -1:]), dim=1)
            elif self.task_emb_cond_type == 'input_add':
                x += task_embedding.unsqueeze(1)
            elif self.task_emb_cond_type == 'cls_token':
                if task_embedding.shape[0] == 1:
                    bs = x.shape[0]
                    task_embedding = task_embedding.expand(bs, -1)
                if self.task_emb_size != self.embedding_dim:
                    task_embedding = self.task_emb_project(task_embedding)
                task_embedding = task_embedding.unsqueeze(1)
                if cls_mask is None:
                    x = torch.cat((task_embedding, x[:, 1:]), dim=1)

        if segment_labels is None:
            segment_labels = torch.zeros_like(tokens)

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if cls_mask is not None:
            if task_embedding is not None:
                x = cls_mask * task_embedding + (1 - cls_mask) * x
            else:
                x = cls_embeddings + (1 - cls_mask) * x

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            # Add extra symbol to account for task embedding
            x *= (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=padding_mask,
                task_emb=task_embedding if self.task_emb_cond_type == 'attention' else None
            )
            if not last_state_only:
                inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        if last_state_only:
            inner_states = [x]

        return inner_states, sentence_rep
