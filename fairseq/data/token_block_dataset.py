# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math

import numpy as np
import torch

from fairseq.data.indexed_dataset import SizedDataset


class TokenBlockDataset(SizedDataset):
    """Given a 1d tensor of tokens, this dataset will break tokens into blocks based on parameters. The blocks are
    fetched from the original tensor so no additional memory is allocated"""

    def __init__(self, tokens, block_size, sizes, offset=0, break_mode=None):
        """
        Args:
            tokens: torch tensor of tokens to break into blocks
            block_size: An integer. the maximum size of each block (note this has no effect in 'eos' break mode)
            sizes: A list of integers. sizes of sentences in the block. the sum of the sizes should add up to the
                   length of tokens
            offset: An integer. rotates the tokens by this much before computing blocks. useful for language model targets
            break_mode: A boolean if None/'none' then breaks tokens into equally sized blocks of size block_size
                        if 'complete' then breaks tokens into block sizes of up to block_size such that each block
                        contains complete sentences. block_size may be exceeded if some sentences exceed block_size
                        if 'eos' then each block contains a single sentence. does not respect block_size"""
        super().__init__()

        self.tokens = tokens
        self.offset = offset
        self.slice_indices = []

        if break_mode is None or break_mode == 'none':
            length = math.ceil(tokens.numel() / block_size)

            def block_at(i):
                start = i * block_size
                end = min(start + block_size, len(tokens))
                return (start, end)

            self.slice_indices = [block_at(i) for i in np.arange(length)]
        elif break_mode == 'complete':
            tok_idx = 0
            sz_idx = 0
            curr_size = 0
            while sz_idx < len(sizes):
                if curr_size + sizes[sz_idx] <= block_size or curr_size == 0:
                    curr_size += sizes[sz_idx]
                    sz_idx += 1
                else:
                    self.slice_indices.append((tok_idx, tok_idx + curr_size))
                    tok_idx += curr_size
                    curr_size = 0
            if curr_size > 0:
                self.slice_indices.append((tok_idx, tok_idx + curr_size))
        elif break_mode == 'eos':
            curr = 0
            for sz in sizes:
                # skip samples with just 1 example (which would be just the eos token)
                if sz > 1:
                    self.slice_indices.append((curr, curr + sz))
                curr += sz
        else:
            raise Exception('invalid break_mode. Supported values: none, complete, eos')

        self._sizes = np.array([e - s for s, e in self.slice_indices])

    def _slice(self, s, e):
        # this will copy only the first block if offset > 0, instead of all blocks if we just rotated
        # the tensor with torch.cat()
        if s < self.offset:
            return torch.cat([self.tokens[s - self.offset:], self.tokens[s:e - self.offset]])
        return self.tokens[s - self.offset:e - self.offset]

    def __getitem__(self, i):
        return self._slice(*self.slice_indices[i])

    def __len__(self):
        return len(self.slice_indices)
