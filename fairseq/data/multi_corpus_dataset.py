# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import OrderedDict
from typing import Dict, List

import numpy as np
from fairseq.data import data_utils

from . import FairseqDataset


logger = logging.getLogger(__name__)


class MultiCorpusDataset(FairseqDataset):
    """
    Stores multiple instances of FairseqDataset together. Requires each instance
    to be the same dataset, as the collate method needs to work on batches with
    samples from each dataset.

    Allows specifying a distribution over the datasets to use. Note that unlike
    MultiCorpusSampledDataset, this distribution allows sampling for each item,
    rather than on a batch level.

    Each time ordered_indices() is called, a new sample is generated with
    the specified distribution.

    Args:
        datasets: a OrderedDict of FairseqDataset instances.
        distribution: a List containing the probability of getting an utterance from
                        corresponding dataset
        seed: random seed for sampling the datsets
        sort_indices: if true, will sort the ordered indices by size
    """

    def __init__(
        self,
        datasets: Dict[str, FairseqDataset],
        distribution: List[float],
        seed: int,
        sort_indices: bool = False,
    ):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        assert len(datasets) == len(distribution)
        self.datasets = datasets
        self.distribution = distribution
        self.seed = seed
        self.sort_indices = sort_indices

        # Avoid repeated conversions to list later
        self.dataset_list = list(datasets.values())
        self.total_num_instances = 0

        first_dataset = list(self.datasets.values())[0]

        self.dataset_offsets = []
        for dataset in datasets.values():
            assert isinstance(dataset, FairseqDataset)
            assert type(dataset) is type(first_dataset)
            self.dataset_offsets.append(self.total_num_instances)
            self.total_num_instances += len(dataset)

    def ordered_indices(self):
        with data_utils.numpy_seed(self.seed, self.epoch):
            # Used to store the order of indices of each dataset to use
            indices = [
                np.random.permutation(len(dataset))
                for dataset in self.datasets.values()
            ]
            # Keep track of which samples we've  used for each dataset
            counters = [0 for _ in self.datasets]

            sampled_indices = [
                self._sample(indices, counters) for _ in range(self.total_num_instances)
            ]
            if self.sort_indices:
                sampled_indices.sort(key=lambda i: self.num_tokens(i))
            return np.array(sampled_indices, dtype=np.int64)

    def _sample(self, indices, counters):
        # First pick dataset
        dataset_idx = np.random.choice(len(self.distribution), p=self.distribution)

        # Then get dataset internal index
        idx = indices[dataset_idx][counters[dataset_idx]]

        # Convert to multi-datasets index
        idx += self.dataset_offsets[dataset_idx]

        counters[dataset_idx] += 1

        # Reset if we reach end
        if counters[dataset_idx] == len(self.dataset_list[dataset_idx]):
            counters[dataset_idx] = 0
            indices[dataset_idx] = np.random.permutation(
                len(self.dataset_list[dataset_idx])
            )

        return idx

    def _map_index(self, index: int):
        """
        If dataset A has length N and dataset B has length M
        then index 1 maps to index 1 of dataset A, and index N + 1
        maps to index 1 of B.
        """
        counter = 0
        for key, dataset in self.datasets.items():
            if index < counter + len(dataset):
                return index - counter, key
            counter += len(dataset)
        raise ValueError(
            "Invalid index: {}, max: {}".format(index, self.total_num_instances)
        )

    def __len__(self):
        """
        Length of this dataset is the sum of individual datasets
        """
        return self.total_num_instances

    def __getitem__(self, index):
        index, key = self._map_index(index)
        return self.datasets[key][index]

    def collater(self, samples):
        """
        Since we enforce all datsets to be the same, collating is just
        picking the first one and doing collate.
        """
        if len(samples) == 0:
            return None

        return list(self.datasets.values())[0].collater(samples)

    def num_tokens(self, index: int):
        index, key = self._map_index(index)
        return self.datasets[key].num_tokens(index)

    def size(self, index: int):
        index, key = self._map_index(index)
        return self.datasets[key].size(index)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @property
    def supports_prefetch(self):
        return False

    @property
    def supports_fetch_outside_dataloader(self):
        return all(
            self.datasets[key].supports_fetch_outside_dataloader
            for key in self.datasets
        )
