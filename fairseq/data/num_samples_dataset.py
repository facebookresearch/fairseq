# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqDataset


class NumSamplesDataset(FairseqDataset):
    def __getitem__(self, index):
        return 1

    def __len__(self):
        return 0

    def collater(self, samples):
        return sum(samples)
