# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import data_utils

from . import BaseWrapperDataset


class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad, pad_length=None):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_length = pad_length

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_length=self.pad_length)


class LeftPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx, pad_length=None):
        super().__init__(dataset, pad_idx, left_pad=True, pad_length=pad_length)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx, pad_length=None):
        super().__init__(dataset, pad_idx, left_pad=False, pad_length=pad_length)
