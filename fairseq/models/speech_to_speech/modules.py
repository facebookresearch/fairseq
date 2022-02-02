# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from fairseq.models import FairseqEncoder
from fairseq.models.transformer import Linear


class CTCDecoder(FairseqEncoder):
    def __init__(self, dictionary, in_dim):
        super().__init__(dictionary)
        self.proj = nn.Linear(in_dim, len(dictionary))

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        encoder_out = self.proj(src_tokens)
        return {"encoder_out": encoder_out}


class StackedEmbedding(nn.Embedding):
    """Embedding module that supports stacked units -> single embedding"""

    def __init__(self, num_embeddings, embed_dim, padding_idx, num_stacked=1):
        super().__init__(num_embeddings, embed_dim, padding_idx)
        # follow transformer.Embedding
        nn.init.normal_(self.weight, mean=0, std=embed_dim**-0.5)
        nn.init.constant_(self.weight[padding_idx], 0)

        self.offset = (
            4  # skip <bos>, <pad>, <eos>, <unk>, specific to fairseq dictionary
        )
        self.vocab_size = num_embeddings - self.offset
        self.num_stacked = num_stacked

        if self.num_stacked > 1:
            self.project_in_dim = Linear(embed_dim * num_stacked, embed_dim, bias=False)

    def forward(self, input):
        if self.num_stacked == 1:
            return super().forward(input)

        # expand input indices
        mask = input >= self.offset
        stacked_input = []
        cum_input = input.new_zeros(input.shape)
        for i in range(1, self.num_stacked + 1):
            div = pow(self.vocab_size, i)
            next_input = torch.remainder(input - self.offset - cum_input, div)
            cum_input += next_input
            next_input = torch.floor_divide(next_input, div // self.vocab_size)
            stacked_input.append((next_input + self.offset) * mask + input * ~mask)

        stacked_input = torch.stack(stacked_input[::-1], dim=2)
        embed = super().forward(stacked_input).view(input.size(0), input.size(1), -1)
        embed = self.project_in_dim(embed)
        return embed
