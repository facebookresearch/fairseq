# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class StripTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, id_to_strip):
        super().__init__(dataset)
        self.id_to_strip = id_to_strip

    def __getitem__(self, index):
        item = self.dataset[index]
        while len(item) > 0 and item[-1] == self.id_to_strip:
            item = item[:-1]
        while len(item) > 0 and item[0] == self.id_to_strip:
            item = item[1:]
        return item
