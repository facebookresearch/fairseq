# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import data_utils
from . import BaseWrapperDataset


class PaddingMaskDataset(BaseWrapperDataset):
    def __init__(self, dataset, left_pad, pad_length=None):
        super().__init__(dataset)
        self.left_pad = left_pad
        self.pad_length = pad_length

    def __getitem__(self, index):
        item = self.dataset[index]
        return torch.zeros_like(item).bool()

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return data_utils.collate_tokens(
            samples, True, left_pad=self.left_pad, pad_to_length=self.pad_length
        )


class LeftPaddingMaskDataset(PaddingMaskDataset):
    def __init__(self, dataset):
        super().__init__(dataset, left_pad=True)


class RightPaddingMaskDataset(PaddingMaskDataset):
    def __init__(self, dataset):
        super().__init__(dataset, left_pad=False)
