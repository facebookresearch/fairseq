# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from fairseq.data import data_utils

from . import BaseWrapperDataset


class TruncateDataset(BaseWrapperDataset):
    """Truncate a sequence by returning the first truncation_length tokens"""

    def __init__(self, dataset, truncation_length):
        super().__init__(dataset)
        assert truncation_length is not None
        self.truncation_length = truncation_length
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        item_len = item.size(0)
        if item_len > self.truncation_length:
            item = item[: self.truncation_length]
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        return len(self.dataset)


class RandomCropDataset(TruncateDataset):
    """Truncate a sequence by returning a random crop of truncation_length tokens"""

    def __init__(self, dataset, truncation_length, seed=1):
        super().__init__(dataset, truncation_length)
        self.seed = seed
        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the crop changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            item_len = item.size(0)
            excess = item_len - self.truncation_length
            if excess > 0:
                start_idx = np.random.randint(0, excess)
                item = item[start_idx : start_idx + self.truncation_length]
            return item


def maybe_shorten_dataset(
    dataset,
    split,
    shorten_data_split_list,
    shorten_method,
    tokens_per_sample,
    seed,
):
    truncate_split = (
        split in shorten_data_split_list.split(",") or len(shorten_data_split_list) == 0
    )
    if shorten_method == "truncate" and truncate_split:
        dataset = TruncateDataset(dataset, tokens_per_sample)
    elif shorten_method == "random_crop" and truncate_split:
        dataset = RandomCropDataset(dataset, tokens_per_sample, seed)
    return dataset
