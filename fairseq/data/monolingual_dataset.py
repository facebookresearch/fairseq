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
            'src_lengths': torch.LongTensor([
                s['source'].numel() for s in samples
            ]),
        },
        'target': merge('target'),
    }


class MonolingualDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
            Default: ``True``
    """

    def __init__(self, dataset, sizes, vocab, shuffle=True):
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
        """Merge a list of samples to form a mini-batch.

        Returns mini-batches with the following keys:
        - `id` (torch.LongTensor): example IDs in the original input order
        - `ntokens` (int): total number of tokens in the batch
        - `net_input` (dict): the input to the Model, containing keys:
          - `src_tokens` (torch.LongTensor): a padded 2D Tensor of tokens in
            the source sentence of shape `(bsz, src_len)`. Padding will appear
            on the right.
        - `target` (torch.LongTensor): a padded 2D Tensor of tokens in the
          target sentence of shape `(bsz, tgt_len)`. Padding will appear on the
          right.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return collate(samples, self.vocab.pad(), self.vocab.eos())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        target = self.vocab.dummy_sentence(tgt_len + 1)
        source, target = target[:-1], target[1:]
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
        order.append(np.flip(self.sizes, 0))
        return np.lexsort(order)
