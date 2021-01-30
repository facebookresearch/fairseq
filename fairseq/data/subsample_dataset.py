# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np

from . import BaseWrapperDataset


logger = logging.getLogger(__name__)


class SubsampleDataset(BaseWrapperDataset):
    """Subsamples a given dataset by a specified ratio. Subsampling is done on the number of examples

    Args:
        dataset (~torch.utils.data.Dataset): dataset to subsample
        size_ratio(float): the ratio to subsample to. must be between 0 and 1 (exclusive)
    """

    def __init__(self, dataset, size_ratio, shuffle=False):
        super().__init__(dataset)
        assert size_ratio < 1
        self.actual_size = np.ceil(len(dataset) * size_ratio).astype(int)
        self.indices = np.random.choice(
            list(range(len(self.dataset))), self.actual_size, replace=False
        )
        self.shuffle = shuffle
        logger.info(
            "subsampled dataset from {} to {} (ratio={})".format(
                len(self.dataset), self.actual_size, size_ratio
            )
        )

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return self.actual_size

    def collater(self, samples):
        return self.dataset.collater(samples)

    @property
    def sizes(self):
        return self.dataset.sizes[self.indices]

    @property
    def name(self):
        return self.dataset.name

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.indices[index])

    def size(self, index):
        return self.dataset.size(self.indices[index])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def prefetch(self, indices):
        self.dataset.prefetch(self.indices[indices])
