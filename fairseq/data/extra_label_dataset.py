# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import FairseqDataset
from typing import List


class ExtraLabelDataset(FairseqDataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, index):
        item = self.dataset[index]
        lbl = self.labels[index]
        item['target'] = [item['target'], torch.LongTensor([lbl])]
        return item

    def __len__(self):
        return len(self.dataset)

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
        return self.dataset.collater(samples)

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        dummy_batch = self.dataset.get_dummy_batch(num_tokens, max_positions, tgt_len)
        dummy_batch['target'] = [dummy_batch['target'],
                                 torch.LongTensor([0 for _ in range(len(dummy_batch['target']))])]
        return dummy_batch

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.dataset.num_tokens(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
