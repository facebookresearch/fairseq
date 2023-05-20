# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np

from . import FairseqDataset, LanguagePairDataset

logger = logging.getLogger(__name__)


class RoundRobinZipDatasets(FairseqDataset):
    """Zip multiple :class:`~fairseq.data.FairseqDataset` instances together.

    Shorter datasets are repeated in a round-robin fashion to match the length
    of the longest one.

    Args:
        datasets (Dict[~fairseq.data.FairseqDataset]): a dictionary of
            :class:`~fairseq.data.FairseqDataset` instances.
        eval_key (str, optional): a key used at evaluation time that causes
            this instance to pass-through batches from *datasets[eval_key]*.
    """

    def __init__(self, datasets, eval_key=None):
        super().__init__()
        if isinstance(datasets, dict):
            datasets = OrderedDict(datasets)
        assert isinstance(datasets, OrderedDict)
        assert datasets, "Can't make a RoundRobinZipDatasets out of nothing"
        for dataset in datasets.values():
            assert isinstance(dataset, FairseqDataset)

        self.datasets = datasets
        self.eval_key = eval_key

        self.longest_dataset_key = max(datasets, key=lambda k: len(datasets[k]))
        self.longest_dataset = datasets[self.longest_dataset_key]
        self._ordered_indices: Dict[str, Sequence[int]] = None

    def _map_index(self, key, index):
        assert (
            self._ordered_indices is not None
        ), "Must call RoundRobinZipDatasets.ordered_indices() first"
        o = self._ordered_indices[key]
        return o[index % len(o)]

    def __getitem__(self, index):
        if self.eval_key is None:
            return OrderedDict(
                [
                    (key, dataset[self._map_index(key, index)])
                    for key, dataset in self.datasets.items()
                ]
            )
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key][self._map_index(self.eval_key, index)]

    def __len__(self):
        if self._ordered_indices is not None:
            return len(self._ordered_indices[self.longest_dataset_key])
        return len(self.longest_dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return None
        if self.eval_key is None:
            return OrderedDict(
                [
                    (key, dataset.collater([sample[key] for sample in samples]))
                    for key, dataset in self.datasets.items()
                ]
            )
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key].collater(samples)

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        # TODO make it configurable whether to use max() or sum() here
        return max(
            dataset.num_tokens(self._map_index(key, index))
            for key, dataset in self.datasets.items()
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return {
            key: dataset.size(self._map_index(key, index))
            for key, dataset in self.datasets.items()
        }

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self._ordered_indices is None:
            # Call the underlying dataset's ordered_indices() here, so that we
            # get the same random ordering as we would have from using the
            # underlying sub-datasets directly.
            self._ordered_indices = OrderedDict(
                [
                    (key, dataset.ordered_indices())
                    for key, dataset in self.datasets.items()
                ]
            )
        return np.arange(len(self))

    def filter_indices_by_size(self, indices, max_positions=None):
        """
        Filter each sub-dataset independently, then update the round robin to work
        on the filtered sub-datasets.
        """

        def _deep_until_language_pair(dataset):
            if isinstance(dataset, LanguagePairDataset):
                return dataset
            if hasattr(dataset, "tgt_dataset"):
                return _deep_until_language_pair(dataset.tgt_dataset)
            if hasattr(dataset, "dataset"):
                return _deep_until_language_pair(dataset.dataset)
            raise Exception(f"Don't know how to unwrap this dataset: {dataset}")

        if not isinstance(max_positions, dict):
            max_positions = {k: max_positions for k in self.datasets.keys()}
        ignored_some = False
        for key, dataset in self.datasets.items():
            dataset = _deep_until_language_pair(dataset)
            self._ordered_indices[key], ignored = dataset.filter_indices_by_size(
                self._ordered_indices[key], max_positions[key]
            )
            if len(ignored) > 0:
                ignored_some = True
                logger.warning(
                    f"{len(ignored)} samples from {key} have invalid sizes and will be skipped, "
                    f"max_positions={max_positions[key]}, first few sample ids={ignored[:10]}"
                )
        # Since we are modifying in place the _ordered_indices,
        # it's not possible anymore to return valid ignored indices.
        # Hopefully the extra debug information print above should be enough to debug.
        # Ideally we would receive ignore_invalid_inputs so that we could have
        # a proper error message.
        return (np.arange(len(self)), [0] if ignored_some else [])

    @property
    def supports_prefetch(self):
        return all(
            getattr(dataset, "supports_prefetch", False)
            for dataset in self.datasets.values()
        )

    def prefetch(self, indices):
        for key, dataset in self.datasets.items():
            dataset.prefetch([self._map_index(key, index) for index in indices])
