# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.data import data_utils

from . import BaseWrapperDataset


class PadDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad)


class LeftPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class RightPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)
