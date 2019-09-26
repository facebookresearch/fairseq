# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset


class ColorizeDataset(BaseWrapperDataset):
    """ Adds 'colors' property to net input that is obtained from the provided color getter for use by models """
    def __init__(self, dataset, color_getter):
        super().__init__(dataset)
        self.color_getter = color_getter

    def collater(self, samples):
        base_collate = super().collater(samples)
        if len(base_collate) > 0:
            base_collate["net_input"]["colors"] = torch.tensor(
                list(self.color_getter(self.dataset, s["id"]) for s in samples),
                dtype=torch.long,
            )
        return base_collate
