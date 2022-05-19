# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.nn.functional as F
from fairseq.data import BaseWrapperDataset
from fairseq.data.data_utils import get_buckets, get_bucketed_sizes


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
        tensor_key=None,
    ):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

        assert num_buckets > 0
        self.buckets = get_buckets(sizes, num_buckets)
        self._bucketed_sizes = get_bucketed_sizes(sizes, self.buckets)
        self._tensor_key = tensor_key

    def _set_tensor(self, item, val):
        if self._tensor_key is None:
            return val
        item[self._tensor_key] = val
        return item

    def _get_tensor(self, item):
        if self._tensor_key is None:
            return item
        return item[self._tensor_key]

    def _pad(self, tensor, bucket_size, dim=-1):
        num_pad = bucket_size - tensor.size(dim)
        return F.pad(
            tensor,
            (num_pad if self.left_pad else 0, 0 if self.left_pad else num_pad),
            value=self.pad_idx,
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        bucket_size = self._bucketed_sizes[index]
        tensor = self._get_tensor(item)
        padded = self._pad(tensor, bucket_size)
        return self._set_tensor(item, padded)

    @property
    def sizes(self):
        return self._bucketed_sizes

    def num_tokens(self, index):
        return self._bucketed_sizes[index]

    def size(self, index):
        return self._bucketed_sizes[index]
