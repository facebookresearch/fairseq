# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad
        self._is_incremental_eval = False

    def incremental_eval(self, mode=True):
        self._is_incremental_eval = mode

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        if self._is_incremental_eval:
            # positions is the same for every token when decoding a single step
            positions = Variable(
                input.data.new(1, 1).fill_(self.padding_idx + input.size(1)))
        else:
            positions = Variable(self.make_positions(input.data))
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1

    def make_positions(self, input):
        """Replace non-padding symbols with their position numbers."""
        if not hasattr(self, 'range_buf'):
            self.range_buf = input.new()
        seqlen = input.size(1)
        if self.range_buf.numel() < seqlen:
            # offset positions by the padding index
            torch.arange(self.padding_idx + 1, self.padding_idx + 1 + seqlen,
                         out=self.range_buf)
        mask = input.ne(self.padding_idx)
        positions = self.range_buf[:seqlen].expand_as(input)
        if self.left_pad:
            positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
        return input.clone().masked_scatter_(mask, positions[mask])
