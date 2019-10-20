# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from . import BaseWrapperDataset


class DynamicTruncateDataset(BaseWrapperDataset):

    def __init__(self, dataset, max_total_length, other_dataset):
        super().__init__(dataset)
        assert max_total_length is not None
        self.max_total_length = max_total_length
        self.dataset = dataset
        self.other_dataset = other_dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        item_len = item.size(0)
        other_item = self.dataset[index]
        other_item_len = other_item.size(0)
        if item_len > (self.max_total_length - other_item_len):
            item = item[:self.max_total_length - other_item_len]
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.max_total_length - self.other_dataset.sizes)

    def __len__(self):
        return len(self.dataset)
