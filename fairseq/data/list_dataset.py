# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class ListDataset(BaseWrapperDataset):
    def __init__(self, dataset, sizes=None):
        super().__init__(dataset)
        self._sizes = sizes

    def __iter__(self):
        for x in self.dataset:
            yield x

    def collater(self, samples):
        return samples

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def set_epoch(self, epoch):
        pass
