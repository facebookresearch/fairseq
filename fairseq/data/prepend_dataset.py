# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class PrependDataset(BaseWrapperDataset):
    def __init__(self, dataset, prepend_getter, ensure_first_token_is=None):
        super().__init__(dataset)
        self.prepend_getter = prepend_getter
        self.ensure_first_token = ensure_first_token_is

    def __getitem__(self, idx):
        item = self.dataset[idx]
        is_tuple = isinstance(item, tuple)
        src = item[0] if is_tuple else item

        assert self.ensure_first_token is None or src[0] == self.ensure_first_token
        prepend_idx = self.prepend_getter(self.dataset, idx)
        assert isinstance(prepend_idx, int)
        src[0] = prepend_idx
        item = tuple((src,) + item[1:]) if is_tuple else src
        return item
