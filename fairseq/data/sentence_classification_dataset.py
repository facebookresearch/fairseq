# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset
from typing import List


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(len(s['sentence']) for s in samples),
        'net_input': {
            'src_tokens': data_utils.collate_tokens(
                [s['sentence'] for s in samples], pad_idx, eos_idx, left_pad=False,
            ),
            'src_lengths': torch.LongTensor([
                s['sentence'].numel() for s in samples
            ]),
        },
        'target': torch.stack([s['target'] for s in samples], dim=0),
        'nsentences': samples[0]['sentence'].size(0),
    }


class SentenceClassificationDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, labels, sizes, dictionary):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.labels = np.array(labels)
        self.vocab = dictionary
        self.shuffle = True

    def __getitem__(self, index):
        sent = self.dataset[index]
        lbl = self.labels[index]
        return {'id': index, 'sentence': sent, 'target': torch.LongTensor([lbl])}

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collate(samples, self.vocab.pad(), self.vocab.eos())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        sent = self.vocab.dummy_sentence(tgt_len + 2)

        return self.collater([
            {'id': i, 'sentence': sent, 'target': torch.LongTensor([0])}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = np.random.permutation(len(self))
            return order
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)


    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.dataset, 'supports_prefetch')
            and self.dataset.supports_prefetch
        )
