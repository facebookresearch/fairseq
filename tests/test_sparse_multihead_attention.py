# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import unittest
from fairseq.modules.sparse_multihead_attention import SparseMultiheadAttention


class TestSparseMultiheadAttention(unittest.TestCase):
    def test_sparse_multihead_attention(self):
        attn_weights = torch.randn(1, 8, 8)
        bidirectional_sparse_mask = torch.tensor([
                [0, 0, 0, 0, 0, float('-inf'), float('-inf'), 0],
                [0, 0, 0, 0, 0, float('-inf'), float('-inf'), 0],
                [0, 0, 0, 0, 0, float('-inf'), float('-inf'), 0],
                [0, 0, 0, 0, 0, float('-inf'), float('-inf'), 0],
                [float('-inf'), float('-inf'), float('-inf'), 0, 0, 0, 0, 0],
                [float('-inf'), float('-inf'), float('-inf'), 0, 0, 0, 0, 0],
                [float('-inf'), float('-inf'), float('-inf'), 0, 0, 0, 0, 0],
                [float('-inf'), float('-inf'), float('-inf'), 0, 0, 0, 0, 0]
            ])

        bidirectional_attention = SparseMultiheadAttention(16, 1, stride=4, expressivity=1, is_bidirectional=True)
        bidirectional_attention_sparse_mask = bidirectional_attention.buffered_sparse_mask(attn_weights, 8, 8)
        torch.all(torch.eq(bidirectional_attention_sparse_mask, bidirectional_sparse_mask))

        sparse_mask = torch.tensor([
                [0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'),
                 float('-inf'), float('-inf')],
                [0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                [0, 0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                [0, 0, 0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                [0, 0, 0, 0, 0, float('-inf'), float('-inf'), float('-inf')],
                [float('-inf'), float('-inf'), float('-inf'), 0, 0, 0, float('-inf'), float('-inf')],
                [float('-inf'), float('-inf'), float('-inf'), 0, 0, 0, 0, float('-inf')],
                [float('-inf'), float('-inf'), float('-inf'), 0, 0, 0, 0, 0],
            ])

        attention = SparseMultiheadAttention(16, 1, stride=4, expressivity=1, is_bidirectional=False)
        attention_sparse_mask = attention.buffered_sparse_mask(attn_weights, 8, 8)

        torch.all(torch.eq(attention_sparse_mask, sparse_mask))


if __name__ == '__main__':
    unittest.main()
