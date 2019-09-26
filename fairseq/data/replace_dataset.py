# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class ReplaceDataset(BaseWrapperDataset):
    """Replaces tokens found in the dataset by a specified replacement token

        Args:
            dataset (~torch.utils.data.Dataset): dataset to replace tokens in
            replace_map(Dictionary[int,int]): map of token to replace -> replacement token
            offsets (List[int]): do not replace tokens before (from left if pos, right if neg) this offset. should be
            as many as the number of objects returned by the underlying dataset __getitem__ method.
        """

    def __init__(self, dataset, replace_map, offsets):
        super().__init__(dataset)
        assert len(replace_map) > 0
        self.replace_map = replace_map
        self.offsets = offsets

    def __getitem__(self, index):
        item = self.dataset[index]
        is_tuple = isinstance(item, tuple)
        srcs = item if is_tuple else [item]

        for offset, src in zip(self.offsets, srcs):
            for k, v in self.replace_map.items():
                src_off = src[offset:] if offset >= 0 else src[:offset]
                src_off.masked_fill_(src_off == k, v)

        item = srcs if is_tuple else srcs[0]
        return item
