# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data.dataloader import default_collate

from . import FairseqDataset


class BaseWrapperDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, "collater"):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        return self.dataset.sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def attr(self, attr: str, index: int):
        return self.dataset.attr(attr, index)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    def get_batch_shapes(self):
        return self.dataset.get_batch_shapes()

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        return self.dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

    def filter_indices_by_size(self, indices, max_sizes):
        return self.dataset.filter_indices_by_size(indices, max_sizes)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return self.dataset.can_reuse_epoch_itr_across_epochs

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
