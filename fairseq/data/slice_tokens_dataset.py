# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from . import BaseWrapperDataset


class SliceTokensDataset(BaseWrapperDataset):
    def __init__(self, dataset, left_slice: Optional[int] = None, right_slice: Optional[int] = None):
        super().__init__(dataset)
        self.left_slice = left_slice
        self.right_slice = right_slice

    def __getitem__(self, index):
        item = self.dataset[index]
        return item[..., self.left_slice:self.right_slice]
