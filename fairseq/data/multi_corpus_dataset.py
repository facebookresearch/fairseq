# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
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
        batch_sample: if true, will ensure each batch is from a single dataset
    """

    def __init__(
        self,
        datasets: Dict[str, FairseqDataset],
        distribution: List[float],
        seed: int,
        sort_indices: bool = False,
        batch_sample: bool = False,
    ):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        assert len(datasets) == len(distribution)
        assert sum(distribution) == 1
        self.datasets = datasets
        self.distribution = distribution
        self.seed = seed
        self.sort_indices = sort_indices
        self.batch_sample = batch_sample

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
        start = time.time()
        with data_utils.numpy_seed(self.seed, self.epoch):
            sampled_indices = []
            num_selected_instances = 0

            # For each dataset i, sample self.distribution[i] * self.total_num_instances
            for i, key in enumerate(self.datasets):

                if i < len(self.datasets) - 1:
                    num_instances = int(self.distribution[i] * self.total_num_instances)
                    high = self.dataset_offsets[i + 1]
                else:
                    num_instances = self.total_num_instances - num_selected_instances
                    high = self.total_num_instances

                logger.info(f"sampling {num_instances} from {key} dataset")
                num_selected_instances += num_instances

                # First, add k copies of the dataset where k = num_instances // len(dataset).
                # This ensures an equal distribution of the data points as much as possible.
                # For the remaining entries randomly sample them
                dataset_size = len(self.datasets[key])
                num_copies = num_instances // dataset_size
                dataset_indices = (
                    np.random.permutation(high - self.dataset_offsets[i])
                    + self.dataset_offsets[i]
                )[: num_instances - num_copies * dataset_size]
                if num_copies > 0:
                    sampled_indices += list(
                        np.concatenate(
                            (
                                np.repeat(
                                    np.arange(self.dataset_offsets[i], high), num_copies
                                ),
                                dataset_indices,
                            )
                        )
                    )
                else:
                    sampled_indices += list(dataset_indices)

            assert (
                len(sampled_indices) == self.total_num_instances
            ), f"{len(sampled_indices)} vs {self.total_num_instances}"

            np.random.shuffle(sampled_indices)
            if self.sort_indices:
                sampled_indices.sort(key=lambda i: self.num_tokens(i))

            logger.info(
                "multi_corpus_dataset ordered_indices took {}s".format(
                    time.time() - start
                )
            )
            return np.array(sampled_indices, dtype=np.int64)

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
        new_index, key = self._map_index(index)
        try:
            item = self.datasets[key][new_index]
            item["full_id"] = index
            return item
        except Exception as e:
            e.args = (f"Error from {key} dataset", *e.args)
            raise

    def collater(self, samples):
        """
        If we are doing batch sampling, then pick the right collater to use.

        Otherwise we assume all collaters are the same.
        """
        if len(samples) == 0:
            return None
        if "full_id" in samples[0]:
            _, key = self._map_index(samples[0]["full_id"])
            return self.datasets[key].collater(samples)
        else:
            # Subclasses may override __getitem__ to not specify full_id
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

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        if not self.batch_sample:
            return super().batch_by_size(
                indices, max_tokens, max_sentences, required_batch_size_multiple
            )

        dataset_indices = {key: [] for key in self.datasets}
        for i in indices:
            _, key = self._map_index(i)
            dataset_indices[key].append(i)

        batches = []
        for key in dataset_indices:
            cur_batches = super().batch_by_size(
                np.array(dataset_indices[key], dtype=np.int64),
                max_tokens,
                max_sentences,
                required_batch_size_multiple,
            )
            logger.info(f"Created {len(cur_batches)} batches for dataset {key}")
            batches += cur_batches

        # Assume shuffling is handled in fairseq/data/iterators.py
        return batches
