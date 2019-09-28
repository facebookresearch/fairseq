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


def collate(samples, pad_idx):
    if len(samples) == 0:
        return {}

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(len(s['sentence']) for s in samples),
        'net_input': {
            'sentence': data_utils.collate_tokens(
                [s['sentence'] for s in samples], pad_idx, left_pad=False,
            ),
            'segment_labels': data_utils.collate_tokens(
                [s['segment_labels'] for s in samples], pad_idx, left_pad=False,
            ),
        },
        'target': torch.stack([s['target'] for s in samples], dim=0),
        'nsentences': len(samples),
    }


class SentencePairClassificationDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset1, dataset2, labels, sizes1, sizes2, dictionary):
        self.dataset1, self.dataset2 = dataset1, dataset2
        self.sizes1, self.sizes2 = np.array(sizes1), np.array(sizes2)
        self.labels = np.array(labels)
        self.vocab = dictionary
        self.shuffle = True
    def __getitem__(self, index):
        sent1 = self.dataset1[index]
        sent2 = self.dataset2[index]
        lbl = self.labels[index]

        sent, segment = self._join_sents(sent1, sent2)
        return {'id': index, 'sentence': sent, 'segment_labels': segment,
                'target': torch.tensor([lbl])}

    def _join_sents(self, sent1, sent2):
        seg0 = torch.zeros(1)
        sent1 = torch.cat([sent1.new(1).fill_(self.vocab.cls()), sent1, sent1.new(1).fill_(self.vocab.sep())])
        seg1 = torch.zeros(sent1.size(0)-1)
        sent2 = torch.cat([sent2, sent2.new(1).fill_(self.vocab.sep())])
        seg2 = torch.ones(sent2.size(0))
        sent = torch.cat([sent1, sent2])
        segment = torch.cat([seg0, seg1, seg2])
        return sent, segment

    def __len__(self):
        return len(self.dataset1)

    def collater(self, samples):
        return collate(samples, self.vocab.pad())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        sent1 = self.vocab.dummy_sentence(tgt_len)
        sent2 = self.vocab.dummy_sentence(tgt_len)

        sent1[sent1.eq(self.vocab.unk())] = 66
        sent2[sent2.eq(self.vocab.unk())] = 66
        sent, seg = self._join_sents(sent1, sent2)

        return self.collater([
            {'id': i, 'sentence': sent, 'segment_labels': seg,
             'target': torch.tensor([self.labels[0]])}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes1[index] + self.sizes2[index] + 3

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes1[index] + self.sizes2[index] + 3

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        indices = indices[np.argsort(self.sizes1[indices], kind='mergesort')]
        return indices[np.argsort(self.sizes2[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.dataset1.prefetch(indices)
        self.dataset2.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
                hasattr(self.dataset1, 'supports_prefetch')
                and self.dataset1.supports_prefetch
                and hasattr(self.dataset2, 'supports_prefetch')
                and self.dataset2.supports_prefetch
        )
