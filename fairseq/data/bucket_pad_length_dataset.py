# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.nn.functional as F
from fairseq.data import BaseWrapperDataset


class BucketPadLengthDataset(BaseWrapperDataset):
    """
    Bucket and pad item lengths to the nearest bucket size. This can be used to
    reduce the number of unique batch shapes, which is important on TPUs since
    each new batch shape requires a recompilation.

    Args:
        dataset (FairseqDatset): dataset to bucket
        sizes (List[int]): all item sizes
        num_buckets (int): number of buckets to create
        pad_idx (int): padding symbol
        left_pad (bool): if True, pad on the left; otherwise right pad
    """

    def __init__(
        self,
        dataset,
        sizes,
        num_buckets,
        pad_idx,
        left_pad,
    ):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

        assert num_buckets > 0
        self.buckets = np.unique(
            np.percentile(
                sizes,
                np.linspace(0, 100, num_buckets + 1),
                interpolation="lower",
            )[1:]
        )

        def get_bucketed_sizes(orig_sizes, buckets):
            sizes = np.copy(orig_sizes)
            assert np.min(sizes) >= 0
            start_val = -1
            for end_val in buckets:
                mask = (sizes > start_val) & (sizes <= end_val)
                sizes[mask] = end_val
                start_val = end_val
            return sizes

        self._bucketed_sizes = get_bucketed_sizes(sizes, self.buckets)

    def __getitem__(self, index):
        item = self.dataset[index]
        bucket_size = self._bucketed_sizes[index]
        num_pad = bucket_size - item.size(-1)
        return F.pad(
            item,
            (num_pad if self.left_pad else 0, 0 if self.left_pad else num_pad),
            value=self.pad_idx,
        )

    @property
    def sizes(self):
        return self._bucketed_sizes

    def num_tokens(self, index):
        return self._bucketed_sizes[index]

    def size(self, index):
        return self._bucketed_sizes[index]
