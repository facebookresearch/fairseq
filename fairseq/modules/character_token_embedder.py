# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils.rnn import pad_sequence

from typing import List, Tuple

from .highway import Highway
from fairseq.data import Dictionary


class CharacterTokenEmbedder(torch.nn.Module):
    def __init__(
            self,
            vocab: Dictionary,
            filters: List[Tuple[int, int]],
            char_embed_dim: int,
            word_embed_dim: int,
            highway_layers: int,
            max_char_len: int = 50,
    ):
        super(CharacterTokenEmbedder, self).__init__()

        self.embedding_dim = word_embed_dim
        self.char_embeddings = nn.Embedding(257, char_embed_dim, padding_idx=0)
        self.symbol_embeddings = nn.Parameter(torch.FloatTensor(2, word_embed_dim))
        self.eos_idx, self.unk_idx = 0, 1

        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(char_embed_dim, out_c, kernel_size=width)
            )

        final_dim = sum(f[1] for f in filters)

        self.highway = Highway(final_dim, highway_layers)
        self.projection = nn.Linear(final_dim, word_embed_dim)

        self.set_vocab(vocab, max_char_len)
        self.reset_parameters()

    def set_vocab(self, vocab, max_char_len):
        word_to_char = torch.LongTensor(len(vocab), max_char_len)

        truncated = 0
        for i in range(len(vocab)):
            if i < vocab.nspecial:
                char_idxs = [0] * max_char_len
            else:
                chars = vocab[i].encode()
                # +1 for padding
                char_idxs = [c + 1 for c in chars] + [0] * (max_char_len - len(chars))
            if len(char_idxs) > max_char_len:
                truncated += 1
                char_idxs = char_idxs[:max_char_len]
            word_to_char[i] = torch.LongTensor(char_idxs)

        if truncated > 0:
            print('Truncated {} words longer than {} characters'.format(truncated, max_char_len))

        self.vocab = vocab
        self.word_to_char = word_to_char

    @property
    def padding_idx(self):
        return self.vocab.pad()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.char_embeddings.weight)
        nn.init.xavier_normal_(self.symbol_embeddings)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.constant_(self.char_embeddings.weight[self.char_embeddings.padding_idx], 0.)
        nn.init.constant_(self.projection.bias, 0.)

    def forward(
            self,
            words: torch.Tensor,
    ):
        self.word_to_char = self.word_to_char.type_as(words)

        flat_words = words.view(-1)
        word_embs = self._convolve(self.word_to_char[flat_words])

        pads = flat_words.eq(self.vocab.pad())
        if pads.any():
            word_embs[pads] = 0

        eos = flat_words.eq(self.vocab.eos())
        if eos.any():
            word_embs[eos] = self.symbol_embeddings[self.eos_idx]

        unk = flat_words.eq(self.vocab.unk())
        if unk.any():
            word_embs[unk] = self.symbol_embeddings[self.unk_idx]

        return word_embs.view(words.size() + (-1,))

    def _convolve(
            self,
            char_idxs: torch.Tensor,
    ):
        char_embs = self.char_embeddings(char_idxs)
        char_embs = char_embs.transpose(1, 2)  # BTC -> BCT

        conv_result = []

        for i, conv in enumerate(self.convolutions):
            x = conv(char_embs)
            x, _ = torch.max(x, -1)
            x = F.relu(x)
            conv_result.append(x)

        conv_result = torch.cat(conv_result, dim=-1)
        conv_result = self.highway(conv_result)

        return self.projection(conv_result)
