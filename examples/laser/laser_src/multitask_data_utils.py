# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict

import numpy as np

from fairseq.data import BaseWrapperDataset, FairseqDataset, iterators


class MultiItr(object):
    def __init__(self, itr, itr_first_batches):
        self.itr = itr
        self._counts = [0 for x in itr]
        self.cur_index = None
        self.itr_first_batches = itr_first_batches

    def __len__(self):
        return sum(len(itr) for itr in self.itr)

    def __iter__(self):
        return self

    def __next__(self, is_dummy=False):
        ratios = [count / len(itr) for count, itr in zip(self._counts, self.itr)]
        idx = ratios.index(min(ratios))
        self.cur_index = idx
        self._counts[idx] += 1
        sample = next(self.itr[idx])
        if len(sample) == 0:
            # use first batch from current iterator
            sample = self.itr_first_batches[idx]
            is_dummy = True
        sample["is_dummy"] = is_dummy
        return sample

    def has_next(self):
        return any([itr.has_next() for itr in self.itr])


class MultidatasetEpochBatchIterator(iterators.EpochBatchIterating):
    """A wrapper around multiple epoch batch iterators."""

    def __init__(
        self,
        dataset,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):

        assert isinstance(dataset, OrderedDict)
        assert len(dataset)
        assert isinstance(dataset[next(iter(dataset))], FairseqDataset)

        self.iterators = []

        self.epoch = epoch
        for key, dt in dataset.items():
            epoch_iter = iterators.EpochBatchIterator(
                dataset=dt,
                collate_fn=dt.collater,
                batch_sampler=batch_sampler[key],
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=0,
                epoch=epoch,
            )
            self.iterators.append(epoch_iter)

    def __len__(self):
        return sum(len(itr) for itr in self.iterators)

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        # `self.epoch += 1` should be handled by underlying `EpochBatchIterator`s.
        self.multi_itr = MultiItr(
            [
                itr.next_epoch_itr(
                    shuffle=shuffle, fix_batches_to_gpus=fix_batches_to_gpus
                )
                for itr in self.iterators
            ],
            [itr.first_batch for itr in self.iterators],
        )
        return self.multi_itr

    def end_of_epoch(self):
        return all(itr.end_of_epoch() for itr in self.iterators)

    @property
    def first_batch(self):
        # return first batch of current iterator
        if hasattr(self, "multi_itr") and self.multi_itr.cur_index:
            return self.iterators[self.multi_itr.cur_index].first_batch
        return self.iterators[0].first_batch

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""

        epochs = [itr.next_epoch_idx for itr in self.iterators]
        self.epoch = epochs[0]
        assert all(epoch == self.epoch for epoch in epochs)

        return self.epoch

    @property
    def iterations_in_epoch(self):
        return sum(itr.iterations_in_epoch for itr in self.iterators)

    def state_dict(self):
        return {
            "iterators": [it.state_dict() for it in self.iterators],
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        for it, d in zip(self.iterators, state_dict["iterators"]):
            it.load_state_dict(d)


class MultitaskDatasetWrapper(BaseWrapperDataset):
    """A wrapper for a multitask dataset."""

    def __init__(
        self, dataset, target_language_id, sample=1.0, do_permutation=False, name=""
    ):
        super().__init__(dataset)
        self.target_language_id = target_language_id
        self.sample = sample
        self.name = name
        self.do_permutation = do_permutation

    def collater(self, *args, **kwargs):
        ans = self.dataset.collater(*args, **kwargs)
        if "net_input" in ans:
            ans["net_input"]["target_language_id"] = self.target_language_id
            ans["net_input"]["dataset_name"] = self.name
        return ans

    def num_tokens(self, *args, **kwargs):
        return self.dataset.num_tokens(*args, **kwargs)

    def ordered_indices(self, *args, **kwargs):
        indices = self.dataset.ordered_indices(*args, **kwargs)
        # Hacky solution for sampling
        repeated_indices = np.repeat(indices, math.ceil(self.sample))

        size = int(self.sample * indices.shape[0])

        permuted = np.random.permutation(repeated_indices.shape[0])[:size]

        if not self.do_permutation:
            permuted = np.sort(permuted)

        return repeated_indices.take(permuted)

    def __len__(self):
        return int(self.sample * len(self.dataset))

    def size(self, index: int):
        return self.dataset.size(index)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
