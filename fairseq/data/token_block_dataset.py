# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import numpy as np
import torch


class TokenBlockDataset(torch.utils.data.Dataset):
    """Break a 1d tensor of tokens into blocks.

    The blocks are fetched from the original tensor so no additional memory is allocated.

    Args:
        tokens: 1d tensor of tokens to break into blocks
        sizes: sentence lengths (required for 'complete' and 'eos')
        block_size: maximum block size (ignored in 'eos' break mode)
        break_mode: Mode used for breaking tokens. Values can be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets: return next tokens as targets
    """

    def __init__(self, tokens, sizes, block_size, break_mode=None, include_targets=False):
        super().__init__()

        self.tokens = tokens
        self.total_size = len(tokens)
        self.include_targets = include_targets
        self.slice_indices = []

        if break_mode is None or break_mode == 'none':
            length = math.ceil(len(tokens) / block_size)

            def block_at(i):
                start = i * block_size
                end = min(start + block_size, len(tokens))
                return (start, end)

            self.slice_indices = [block_at(i) for i in range(length)]
        elif break_mode == 'complete':
            assert sizes is not None and sum(sizes) == len(tokens)
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
            assert sizes is not None and sum(sizes) == len(tokens)
            curr = 0
            for sz in sizes:
                # skip samples with just 1 example (which would be just the eos token)
                if sz > 1:
                    self.slice_indices.append((curr, curr + sz))
                curr += sz
        else:
            raise ValueError('Invalid break_mode: ' + break_mode)

        self.sizes = np.array([e - s for s, e in self.slice_indices])

    def __getitem__(self, index):
        s, e = self.slice_indices[index]
        item = torch.LongTensor(self.tokens[s:e])
        if self.include_targets:
            if e == self.total_size:
                return item[:-1], item[1:]
            else:
                return item, torch.LongTensor(self.tokens[s + 1:e + 1])
        else:
            return item

    def __len__(self):
        return len(self.slice_indices)
