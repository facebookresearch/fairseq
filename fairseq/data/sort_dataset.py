# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
