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

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(data_utils.collate_tokens(
                    [s[key][i] for s in samples], pad_idx, eos_idx, left_pad=False,
                ))
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, eos_idx, left_pad=False,
            )

    is_target_list = isinstance(samples[0]['target'], list)

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(len(s['source']) for s in samples),
        'net_input': {
            'src_tokens': merge('source'),
            'src_lengths': torch.LongTensor([
                s['source'].numel() for s in samples
            ]),
        },
        'target': merge('target', is_target_list),
        'nsentences': samples[0]['source'].size(0),
    }


class MonolingualDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, sizes, src_vocab, tgt_vocab, add_eos_for_other_targets, shuffle,
                 targets=None):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle

        assert targets is None or all(
            t in {'self', 'future', 'past'} for t in targets), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        self.targets = targets

    def __getitem__(self, index):
        source, future_target, past_target = self.dataset[index]
        source, target = self._make_source_target(source, future_target, past_target)
        return {'id': index, 'source': source, 'target': target}

    def __len__(self):
        return len(self.dataset)

    def _make_source_target(self, source, future_target, past_target):
        if self.targets is not None:
            target = []

            if self.add_eos_for_other_targets and (('self' in self.targets) or ('past' in self.targets)) \
                    and source[-1] != self.vocab.eos():
                # append eos at the end of source
                source = torch.cat([source, source.new([self.vocab.eos()])])

                if 'future' in self.targets:
                    future_target = torch.cat([future_target, future_target.new([self.vocab.pad()])])
                if 'past' in self.targets:
                    # first token is before the start of sentence which is only used in "none" break mode when
                    # add_eos_for_other_targets is False
                    past_target = torch.cat([past_target.new([self.vocab.pad()]), past_target[1:], source[-2, None]])

            for t in self.targets:
                if t == 'self':
                    target.append(source)
                elif t == 'future':
                    target.append(future_target)
                elif t == 'past':
                    target.append(past_target)
                else:
                    raise Exception('invalid target ' + t)

            if len(target) == 1:
                target = target[0]
        else:
            target = future_target

        return source, self._filter_vocab(target)

    def _filter_vocab(self, target):
        if len(self.tgt_vocab) != len(self.vocab):
            def _filter(target):
                mask = target.ge(len(self.tgt_vocab))
                if mask.any():
                    target[mask] = self.tgt_vocab.unk()
                return target

            if isinstance(target, list):
                return [_filter(t) for t in target]
            return _filter(target)
        return target

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(samples, self.vocab.pad(), self.vocab.eos())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = max(num_tokens // tgt_len, 1)
        target = self.vocab.dummy_sentence(tgt_len + 2)
        source, past_target, future_target = target[1:-1], target[2:], target[:-2]
        source, target = self._make_source_target(source, past_target, future_target)

        return self.collater([
            {'id': i, 'source': source, 'target': target}
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
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)
