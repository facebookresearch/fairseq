# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import BaseWrapperDataset, data_utils


class AddClassTargetDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        multi_class,
        num_classes=None,
        label_indices=None,
        add_to_input=True,
    ):
        super().__init__(dataset)

        self.label_indices = label_indices
        self.labels = labels
        self.multi_class = multi_class
        self.add_to_input = add_to_input
        if num_classes is None and multi_class:
            assert self.label_indices is not None
            num_classes = len(self.label_indices)

        self.num_classes = num_classes

    def __getitem__(self, index):
        item = self.dataset[index]
        item_labels = self.labels[index]
        if self.multi_class:
            item["label"] = torch.zeros(self.num_classes)
            for il in item_labels:
                if self.label_indices is not None:
                    il = self.label_indices[il]
                item["label"][il] = 1.0
        else:
            item["label"] = torch.tensor(
                self.labels[index]
                if self.label_indices is None
                else self.label_indices[self.labels[index]]
            )

        return item

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated

        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]
        collated["label"] = torch.stack(target, dim=0)

        if self.add_to_input:
            collated["net_input"]["label"] = collated["label"]

        return collated
