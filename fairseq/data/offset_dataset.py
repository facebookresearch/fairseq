# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


from torch.utils.data import Dataset


class OffsetDataset(Dataset):
    """ Wraps an existing dataset, but starts iterating from a particular offset """

    def __init__(self, dataset, offset):
        """
        Args:
            dataset: Dataset to wrap
            offset: An integer. offset from which to start iterating
        """
        super().__init__()

        assert len(dataset) >= offset

        self.dataset = dataset
        self.offset = offset

    def __getitem__(self, i):
        return self.dataset[i + self.offset]

    def __len__(self):
        return len(self.dataset) - self.offset
