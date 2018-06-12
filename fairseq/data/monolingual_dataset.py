# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples], pad_idx, eos_idx, left_pad=False,
        )

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(len(s['target']) for s in samples),
        'net_input': {
            'src_tokens': merge('source'),
        },
        'target': merge('target'),
    }


class MonolingualDataset(FairseqDataset):
    """A wrapper around torch.utils.data.Dataset for monolingual data."""

    def __init__(self, dataset, sizes, vocab, shuffle):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.shuffle = shuffle

    def __getitem__(self, index):
        source, target = self.dataset[index]
        return {'id': index, 'source': source, 'target': target}

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(samples, self.vocab.pad(), self.vocab.eos())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        assert isinstance(max_positions, float) or isinstance(max_positions, int)
        tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        target = self.vocab.dummy_sentence(tgt_len + 1)
        source, target = target[:-1], target[1:]
        return self.collater([
            {'id': i, 'source': source, 'target': target}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        source, target = self.dataset[index]
        return len(source)

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        assert isinstance(max_positions, float) or isinstance(max_positions, int)
        return self.sizes[index] <= max_positions
