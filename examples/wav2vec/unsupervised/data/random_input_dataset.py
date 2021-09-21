# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List

from fairseq.data import BaseWrapperDataset, data_utils


class RandomInputDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        random_input_dataset,
        input_key_path: List[str],
        add_to_input,
        pad_idx,
    ):
        super().__init__(dataset)
        self.random_input_dataset = random_input_dataset
        if isinstance(input_key_path, str):
            input_key_path = [input_key_path]
        assert len(input_key_path) > 0
        self.input_key_path = input_key_path
        self.add_to_input = add_to_input
        self.pad_idx = pad_idx

    def get_target(self, item):
        target_loc = item
        for p in self.input_key_path[:-1]:
            target_loc = target_loc[p]
        return self.input_key_path[-1], target_loc

    def get_target_value(self, item):
        k, target_loc = self.get_target(item)
        return target_loc[k]

    def __getitem__(self, index):
        item = self.dataset[index]
        k, target_loc = self.get_target(item)
        target_loc[k] = random.choice(self.random_input_dataset)
        return item

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())

        random_inputs = data_utils.collate_tokens(
            [self.get_target_value(s) for s in samples if s["id"] in indices],
            pad_idx=self.pad_idx,
            left_pad=False,
        )
        k, target_loc = self.get_target(
            collated if not self.add_to_input else collated["net_input"]
        )
        target_loc[k] = random_inputs

        return collated
