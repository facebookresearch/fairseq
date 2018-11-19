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

    target_len = len(samples[0]['target'])
    target = [torch.stack([s['target'][i] for s in samples], dim=0) for i in range(target_len)]

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(len(s['text']) for s in samples),
        'net_input': {
            'text': data_utils.collate_tokens(
                [s['text'] for s in samples], pad_idx, eos_idx, left_pad=False,
            ),
            'paragraph_mask': data_utils.collate_tokens([s['paragraph_mask'] for s in samples], 0, 0, left_pad=False)
        },
        'target': target,
        'nsentences': len(samples),
        'possible_sentences': sum(int(s['target'][0] == 0) for s in samples),
    }


class SquadDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset1, dataset2, labels, sizes1, sizes2, dictionary, pad_idx, concat_sentences_mode):
        self.dataset1, self.dataset2 = dataset1, dataset2
        self.sizes1, self.sizes2 = np.array(sizes1), np.array(sizes2)
        self.labels = np.array(labels)
        self.vocab = dictionary
        self.shuffle = True
        self.pad_idx = pad_idx
        self.concat_sentences_mode = concat_sentences_mode

    def __getitem__(self, index):
        paragraph = self.dataset1[index]
        question = self.dataset2[index]
        lbl = self.labels[index]

        text = self._join_sents(question, paragraph)

        question_len = question.numel() + 1  # account for bos

        paragraph_mask = torch.zeros(text.shape).byte()
        start_target = torch.LongTensor(1)

        if len(lbl) == 0:
            is_impossible_target = torch.tensor([1])
            start_target.fill_(self.pad_idx)
            end_target = start_target
        else:
            is_impossible_target = torch.tensor([0])
            paragraph_mask[question_len:] = 1  # include last eos in case it is the end index
            end_target = start_target.clone()

            s, e = lbl[0]
            assert e > s
            start_target.fill_(question_len + s)
            end_target.fill_(question_len + e)

        target = (is_impossible_target, start_target, end_target)

        return {'id': index, 'text': text, 'target': target, 'paragraph_mask': paragraph_mask}

    def _join_sents(self, sent1, sent2):
        eos = sent1.new_full((1,), self.vocab.eos())
        sent1 = torch.cat([eos, sent1])

        if self.concat_sentences_mode == 'eos':
            text = torch.cat([sent1, sent2])
        elif self.concat_sentences_mode == 'unk':
            text = torch.cat([sent1, sent1.new_full((1,), self.vocab.unk()), eos, sent2])
        elif self.concat_sentences_mode == 'unk_only':
            text = torch.cat([sent1[:-1], sent1.new_full((1,), self.vocab.unk()), sent2])
        else:
            raise Exception('unknown concat sentence mode ' + self.concat_sentences_mode)

        return text

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
        sent2 = self.vocab.dummy_sentence(tgt_len + 2)

        sent1[sent1.eq(self.vocab.unk())] = 66
        sent2[sent2.eq(self.vocab.unk())] = 66
        text = self._join_sents(sent1, sent2)

        paragraph_mask = torch.zeros(text.shape).byte()
        paragraph_mask[sent2.numel():] = 1

        target = (torch.tensor([0]), torch.tensor([self.pad_idx]), torch.tensor([self.pad_idx]))

        return self.collater([
            {'id': i, 'text': text, 'target': target, 'paragraph_mask': paragraph_mask}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes1[index] + self.sizes2[index] + 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes1[index] + self.sizes2[index] + 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            indices = np.random.permutation(len(self))
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
