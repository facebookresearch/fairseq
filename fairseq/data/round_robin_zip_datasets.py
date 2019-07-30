# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import numpy as np

from . import FairseqDataset


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
        assert isinstance(datasets, OrderedDict)
        self.datasets = datasets
        self.eval_key = eval_key

        self.longest_dataset = None
        self.longest_dataset_key = None
        for key, dataset in datasets.items():
            assert isinstance(dataset, FairseqDataset)
            if self.longest_dataset is None or len(dataset) > len(self.longest_dataset):
                self.longest_dataset = dataset
                self.longest_dataset_key = key

        self._ordered_indices = None

    def _map_index(self, key, index):
        assert self._ordered_indices is not None, \
            'Must call RoundRobinZipDatasets.ordered_indices() first'
        return self._ordered_indices[key][index % len(self.datasets[key])]

    def __getitem__(self, index):
        if self.eval_key is None:
            return OrderedDict([
                (key, dataset[self._map_index(key, index)])
                for key, dataset in self.datasets.items()
            ])
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key][self._map_index(self.eval_key, index)]

    def __len__(self):
        return len(self.longest_dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return None
        if self.eval_key is None:
            return OrderedDict([
                (key, dataset.collater([sample[key] for sample in samples]))
                for key, dataset in self.datasets.items()
            ])
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
            # underlying dataset directly.
            self._ordered_indices = OrderedDict([
                (key, dataset.ordered_indices())
                for key, dataset in self.datasets.items()
            ])
        return np.arange(len(self))

    @property
    def supports_prefetch(self):
        return all(
            getattr(dataset, 'supports_prefetch', False)
            for dataset in self.datasets.values()
        )

    def prefetch(self, indices):
        for key, dataset in self.datasets.items():
            dataset.prefetch([self._map_index(key, index) for index in indices])
