# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import FairseqDataset


class AssertSameLengthDataset(FairseqDataset):

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __getitem__(self, index):
        assert torch.numel(self.first[index]) == torch.numel(self.second[index])

    def __len__(self):
        return 0

    def collater(self, samples):
        return 0
