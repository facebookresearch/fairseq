# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from fairseq.data import Dictionary

from .highway import Highway

CHAR_PAD_IDX = 0
CHAR_EOS_IDX = 257


logger = logging.getLogger(__name__)


class CharacterTokenEmbedder(torch.nn.Module):
    def __init__(
            self,
            vocab: Dictionary,
            filters: List[Tuple[int, int]],
            char_embed_dim: int,
            word_embed_dim: int,
            highway_layers: int,
            max_char_len: int = 50,
            char_inputs: bool = False
    ):
        super(CharacterTokenEmbedder, self).__init__()

        self.onnx_trace = False
        self.embedding_dim = word_embed_dim
        self.max_char_len = max_char_len
        self.char_embeddings = nn.Embedding(257, char_embed_dim, padding_idx=0)
        self.symbol_embeddings = nn.Parameter(torch.FloatTensor(2, word_embed_dim))
        self.eos_idx, self.unk_idx = 0, 1
        self.char_inputs = char_inputs

        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(char_embed_dim, out_c, kernel_size=width)
            )

        last_dim = sum(f[1] for f in filters)

        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None

        self.projection = nn.Linear(last_dim, word_embed_dim)

        assert vocab is not None or char_inputs, "vocab must be set if not using char inputs"
        self.vocab = None
        if vocab is not None:
            self.set_vocab(vocab, max_char_len)

        self.reset_parameters()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

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
            logger.info('truncated {} words longer than {} characters'.format(truncated, max_char_len))

        self.vocab = vocab
        self.word_to_char = word_to_char

    @property
    def padding_idx(self):
        return Dictionary().pad() if self.vocab is None else self.vocab.pad()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.char_embeddings.weight)
        nn.init.xavier_normal_(self.symbol_embeddings)
        nn.init.xavier_uniform_(self.projection.weight)

        nn.init.constant_(self.char_embeddings.weight[self.char_embeddings.padding_idx], 0.)
        nn.init.constant_(self.projection.bias, 0.)

    def forward(
            self,
            input: torch.Tensor,
    ):
        if self.char_inputs:
            chars = input.view(-1, self.max_char_len)
            pads = chars[:, 0].eq(CHAR_PAD_IDX)
            eos = chars[:, 0].eq(CHAR_EOS_IDX)
            if eos.any():
                if self.onnx_trace:
                    chars = torch.where(eos.unsqueeze(1), chars.new_zeros(1), chars)
                else:
                    chars[eos] = 0

            unk = None
        else:
            flat_words = input.view(-1)
            chars = self.word_to_char[flat_words.type_as(self.word_to_char)].type_as(input)
            pads = flat_words.eq(self.vocab.pad())
            eos = flat_words.eq(self.vocab.eos())
            unk = flat_words.eq(self.vocab.unk())

        word_embs = self._convolve(chars)
        if self.onnx_trace:
            if pads.any():
                word_embs = torch.where(pads.unsqueeze(1), word_embs.new_zeros(1), word_embs)
            if eos.any():
                word_embs = torch.where(eos.unsqueeze(1), self.symbol_embeddings[self.eos_idx], word_embs)
            if unk is not None and unk.any():
                word_embs = torch.where(unk.unsqueeze(1), self.symbol_embeddings[self.unk_idx], word_embs)
        else:
            if pads.any():
                word_embs[pads] = 0
            if eos.any():
                word_embs[eos] = self.symbol_embeddings[self.eos_idx]
            if unk is not None and unk.any():
                word_embs[unk] = self.symbol_embeddings[self.unk_idx]

        return word_embs.view(input.size()[:2] + (-1,))

    def _convolve(
            self,
            char_idxs: torch.Tensor,
    ):
        char_embs = self.char_embeddings(char_idxs)
        char_embs = char_embs.transpose(1, 2)  # BTC -> BCT

        conv_result = []

        for conv in self.convolutions:
            x = conv(char_embs)
            x, _ = torch.max(x, -1)
            x = F.relu(x)
            conv_result.append(x)

        x = torch.cat(conv_result, dim=-1)

        if self.highway is not None:
            x = self.highway(x)
        x = self.projection(x)

        return x
