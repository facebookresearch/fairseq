# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import BaseWrapperDataset


class OffsetTokensDataset(BaseWrapperDataset):

    def __init__(self, dataset, offset):
        super().__init__(dataset)
        self.offset = offset

    def __getitem__(self, idx):
        return self.dataset[idx] + self.offset
