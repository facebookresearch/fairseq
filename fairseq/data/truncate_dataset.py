# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np

from . import BaseWrapperDataset


class TruncateDataset(BaseWrapperDataset):

    def __init__(self, dataset, truncation_length):
        super().__init__(dataset)
        self.truncation_length = truncation_length
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        item_len = item.size(0)
        if item_len > self.truncation_length:
            item = item[:self.truncation_length]
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        return len(self.dataset)
