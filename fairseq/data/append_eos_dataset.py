# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch


class AppendEosDataset(torch.utils.data.Dataset):
    """A dataset wrapper that appends EOS to each item."""

    def __init__(self, dataset, eos):
        self.dataset = dataset
        self.eos = eos

    def __getitem__(self, index):
        item = torch.cat([self.dataset[index], torch.LongTensor([self.eos])])
        print(item)
        return item

    def __len__(self):
        return len(self.dataset)
