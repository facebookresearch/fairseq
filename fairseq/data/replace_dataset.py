# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class ReplaceDataset(BaseWrapperDataset):
    def __init__(self, dataset, replace_map, offset=0):
        super().__init__(dataset)
        assert len(replace_map) > 0
        self.replace_map = replace_map
        self.offset = offset

    def __getitem__(self, index):
        item = self.dataset[index]
        is_tuple = isinstance(item, tuple)
        src = item[0] if is_tuple else item

        for k, v in self.replace_map.items():
            src_off = src[self.offset:]
            src_off.masked_fill_(src_off == k, v)

        item = tuple((src,) + item[1:]) if is_tuple else src
        return item
