# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import FairseqDataset


class IdDataset(FairseqDataset):

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 0

    def collater(self, samples):
        return torch.tensor(samples)
