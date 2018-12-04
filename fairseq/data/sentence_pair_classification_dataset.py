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
        'ntokens': sum(len(s['sentence1'] + len(s['sentence2'])) for s in samples),
        'net_input': {
            'sentence1': data_utils.collate_tokens(
                [s['sentence1'] for s in samples], pad_idx, eos_idx, left_pad=False,
            ),
            'sentence2': data_utils.collate_tokens(
                [s['sentence2'] for s in samples], pad_idx, eos_idx, left_pad=False,
            ),
            'sent1_lengths': torch.stack([s['sent1_lengths'] for s in samples], dim=0),
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

    def __init__(self, dataset1, dataset2, labels, sizes1, sizes2, dictionary, concat_sentences_mode):
        self.dataset1, self.dataset2 = dataset1, dataset2
        self.sizes1, self.sizes2 = np.array(sizes1), np.array(sizes2)
        self.labels = np.array(labels)
        self.vocab = dictionary
        self.shuffle = True
        self.concat_sentences_mode = concat_sentences_mode

    def __getitem__(self, index):
        sent1 = self.dataset1[index]
        sent1_len = sent1.numel()
        sent2 = self.dataset2[index]
        lbl = self.labels[index]

        sent1, sent2 = self._join_sents(sent1, sent2)

        return {'id': index, 'sentence1': sent1, 'sentence2': sent2, 'sent1_lengths': torch.LongTensor([sent1_len]),
                'target': torch.tensor([lbl])}

    def _join_sents(self, sent1, sent2):
        eos = sent1.new_full((1,), self.vocab.eos())
        sent1 = torch.cat([eos, sent1])

        if self.concat_sentences_mode == 'none':
            sent2 = torch.cat([eos, sent2])
        elif self.concat_sentences_mode == 'eos':
            sent1 = torch.cat([sent1, sent2])
            sent2 = sent2.new(0)
        elif self.concat_sentences_mode == 'unk':
            sent1 = torch.cat([sent1, sent1.new_full((1,), self.vocab.unk()), eos, sent2])
            sent2 = sent2.new(0)
        elif self.concat_sentences_mode == 'unk_only':
            sent1 = torch.cat([sent1[:-1], sent1.new_full((1,), self.vocab.unk()), sent2])
            sent2 = sent2.new(0)
        elif self.concat_sentences_mode == 'fixed':
            size = 50
            org_size = sent1.numel()
            sent1.resize_(size)
            sent1[min(org_size - 1, size - 1)] = self.vocab.eos()
            sent1[org_size:] = self.vocab.unk()
            sent1 = torch.cat([sent1, eos, sent2])
            sent2 = sent2.new(0)
        elif self.concat_sentences_mode == 'sep':
            sent1 = torch.cat([sent1[:-1], sent1.new_full((1,), len(self.vocab) - 1), sent2])
            sent2 = sent2.new(0)
        else:
            raise Exception('unknown concat sentence mode ' + self.concat_sentences_mode)
        return sent1, sent2

    def __len__(self):
        return len(self.dataset1)

    def collater(self, samples):
        return collate(samples, self.vocab.pad(), self.vocab.eos())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        sent1 = self.vocab.dummy_sentence(tgt_len + 2)
        sent1_len = sent1.numel()
        sent2 = self.vocab.dummy_sentence(tgt_len + 2)

        sent1[sent1.eq(self.vocab.unk())] = 66
        sent2[sent2.eq(self.vocab.unk())] = 66
        sent1, sent2 = self._join_sents(sent1, sent2)

        return self.collater([
            {'id': i, 'sentence1': sent1, 'sentence2': sent2, 'sent1_lengths': torch.LongTensor([sent1_len]),
             'target': torch.tensor([self.labels[0]])}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes1[index] + self.sizes2[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes1[index] + self.sizes2[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            indices = np.random.permutation(len(self))
            return indices
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
