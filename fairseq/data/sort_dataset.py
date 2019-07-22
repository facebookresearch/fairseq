# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np

from . import BaseWrapperDataset


class SortDataset(BaseWrapperDataset):

    def __init__(self, dataset, sort_order):
        super().__init__(dataset)
        if not isinstance(sort_order, (list, tuple)):
            sort_order = [sort_order]
        self.sort_order = sort_order

        assert all(len(so) == len(dataset) for so in sort_order)

    def ordered_indices(self):
        return np.lexsort(self.sort_order)
