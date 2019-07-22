# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import BaseWrapperDataset


class NumelDataset(BaseWrapperDataset):

    def __init__(self, dataset, reduce=False):
        super().__init__(dataset)
        self.reduce = reduce

    def __getitem__(self, index):
        item = self.dataset[index]
        if torch.is_tensor(item):
            return torch.numel(item)
        else:
            return np.size(item)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.reduce:
            return sum(samples)
        else:
            return torch.tensor(samples)
