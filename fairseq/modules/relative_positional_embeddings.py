# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn

from torch.autograd import Variable

class RelativePositionalEmbedding(nn.Embedding):
    """This module learns relative positional embeddings up to a fixed maximum size (k).

    The implementation follows the paper "Self-Attention with Relative Position Representations" (Shaw, 2018)

    Given a length of tokens n, generates an n x n x d tensor that contains embeddings for each pair of tokens
    """

    def __init__(self, k, embedding_dim):
        super().__init__(k*2+1, embedding_dim)
        self.k = k

    def forward(self, length):
        x = torch.arange(length, out=self.weight.new().long()).expand(length, length)
        # make each cell index to relative position
        x = x - x.t()
        x = x.clamp(-self.k, self.k)
        x = x + self.k

        return super().forward(Variable(x))
