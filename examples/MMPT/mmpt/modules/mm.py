# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Facebook, Inc. All Rights Reserved


import torch

from torch import nn

try:
    from transformers.modeling_bert import (
        BertEmbeddings,
        ACT2FN,
    )
except ImportError:
    pass


class VideoTokenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.input_dim if hasattr(config, "input_dim") else 512
        self.linear1 = nn.Linear(input_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MMBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.max_video_len = config.max_video_len
        if hasattr(config, "use_seg_emb") and config.use_seg_emb:
            """the original VLM paper uses seg_embeddings for temporal space.
            although not used it changed the randomness of initialization.
            we keep it for reproducibility.
            """
            self.seg_embeddings = nn.Embedding(256, config.hidden_size)

    def forward(
        self,
        input_ids,
        input_video_embeds,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        input_tensor = input_ids if input_ids is not None else inputs_embeds
        if input_video_embeds is not None:
            input_shape = (
                input_tensor.size(0),
                input_tensor.size(1) + input_video_embeds.size(1),
            )
        else:
            input_shape = (input_tensor.size(0), input_tensor.size(1))

        if position_ids is None:
            """
            Auto skip position embeddings for text only case.
            use cases:
            (1) action localization and segmentation:
                feed in len-1 dummy video token needs text part to
                skip input_video_embeds.size(1) for the right
                position_ids for video [SEP] and rest text tokens.
            (2) MMFusionShare for two forward passings:
                in `forward_text`: input_video_embeds is None.
                    need to skip video [SEP] token.

            # video_len + 1: [CLS] + video_embed
            # self.max_video_len + 1: [SEP] for video.
            # self.max_video_len + 2: [SEP] for video.
            # self.max_video_len + input_ids.size(1): rest for text.
            """
            if input_video_embeds is not None:
                video_len = input_video_embeds.size(1)
                starting_offset = self.max_video_len + 1  # video [SEP]
                ending_offset = self.max_video_len + input_ids.size(1)
            else:
                video_len = 0
                starting_offset = self.max_video_len + 2  # first text token.
                ending_offset = self.max_video_len + input_ids.size(1) + 1
            position_ids = torch.cat([
                self.position_ids[:, :video_len + 1],
                self.position_ids[:, starting_offset:ending_offset]
                ], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        """
        the format of input_ids is [CLS] [SEP] caption [SEP] padding.
        the goal is to build [CLS] video tokens [SEP] caption [SEP] .
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if input_video_embeds is not None:
            inputs_mm_embeds = torch.cat([
                inputs_embeds[:, :1], input_video_embeds, inputs_embeds[:, 1:]
            ], dim=1)
        else:
            # text only for `MMFusionShare`.
            inputs_mm_embeds = inputs_embeds

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_mm_embeds + position_embeddings
        embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlignHead(nn.Module):
    """this will load pre-trained weights for NSP, which is desirable."""

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, dropout_pooled_output):
        logits = self.seq_relationship(dropout_pooled_output)
        return logits
