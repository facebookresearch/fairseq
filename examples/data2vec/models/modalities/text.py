# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq.modules import PositionalEmbedding, FairseqDropout, LayerNorm
from fairseq.tasks import FairseqTask
from .base import D2vModalityConfig, ModalitySpecificEncoder, get_alibi_bias
from .modules import BlockEncoder, Decoder1d
from examples.data2vec.data.modality import Modality


@dataclass
class D2vTextConfig(D2vModalityConfig):
    type: Modality = Modality.TEXT
    max_source_positions: int = 512
    learned_pos: bool = True
    dropout: float = 0.1  # used for both local_encoder and contextualized encoder. tied with global transformer in data2vec_text

    no_scale_embedding: bool = True
    layernorm_embedding: bool = True
    no_token_positional_embeddings: bool = False


class TextEncoder(ModalitySpecificEncoder):

    modality_cfg: D2vTextConfig

    def __init__(
        self,
        modality_cfg: D2vTextConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
    ):
        self.pad_idx = task.source_dictionary.pad()
        self.vocab_size = len(task.source_dictionary)

        local_encoder = TextLocalEncoder(
            vocab_size=self.vocab_size,
            embed_dim=embed_dim,
            max_source_positions=modality_cfg.max_source_positions,
            pad_idx=self.pad_idx,
            no_scale_embedding=modality_cfg.no_scale_embedding,
            layernorm_embedding=modality_cfg.layernorm_embedding,
            dropout=modality_cfg.dropout,
            no_token_positional_embeddings=modality_cfg.no_token_positional_embeddings,
            learned_pos=modality_cfg.learned_pos,
        )
        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim)
            if not layer_norm_first and modality_cfg.prenet_depth > 0
            else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout if modality_cfg.prenet_depth > 0 else 0.0,
        )
        decoder = (
            Decoder1d(modality_cfg.decoder, embed_dim)
            if modality_cfg.decoder is not None
            else None
        )

        alibi_bias_fn = partial(get_alibi_bias, alibi_biases=alibi_biases)

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=nn.Identity(),
            fixed_positional_encoder=None,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def reset_parameters(self):
        super().reset_parameters()

    def convert_padding_mask(self, x, padding_mask):
        if padding_mask is None or padding_mask.size(1) == x.size(1):
            return padding_mask

        diff = self.downsample - padding_mask.size(1) % self.downsample
        if 0 < diff < self.downsample:
            padding_mask = F.pad(padding_mask, (0, diff), value=True)

        padding_mask = padding_mask.view(padding_mask.size(0), -1, self.downsample)
        padding_mask = padding_mask.all(-1)
        if padding_mask.size(1) > x.size(1):
            padding_mask = padding_mask[:, : x.size(1)]

        assert x.size(1) == padding_mask.size(
            1
        ), f"{x.size(1), padding_mask.size(1), diff, self.downsample}"

        return padding_mask


class TextLocalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_source_positions,
        pad_idx,
        no_scale_embedding,
        layernorm_embedding,
        dropout,
        no_token_positional_embeddings,
        learned_pos,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.dropout_module = FairseqDropout(dropout)

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                max_source_positions,
                embed_dim,
                pad_idx,
                learned=learned_pos,
            )
            if not no_token_positional_embeddings
            else None
        )
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embed_dim)

        self.layernorm_embedding = None
        if layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)

    def forward(self, src_tokens):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x
