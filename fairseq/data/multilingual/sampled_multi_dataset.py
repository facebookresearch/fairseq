# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from enum import Enum
from collections import OrderedDict
from collections import defaultdict
from bisect import bisect_right
import hashlib
import logging
import datetime
import time

import numpy as np
import torch

from fairseq import distributed_utils
from fairseq.data import plasma_utils, FairseqDataset


def get_time_gap(s, e):
    return (datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)).__str__()


logger = logging.getLogger(__name__)


def default_virtual_size_func(datasets, ratios, max_scale_up=1.5):
    sizes = [len(d) for d in datasets]
    if ratios is None:
        return sum(sizes)
    largest_idx = np.argmax(sizes)
    largest_r = ratios[largest_idx]
    largest_s = sizes[largest_idx]
    # set virtual sizes relative to the largest dataset
    virtual_sizes = [(r / largest_r) * largest_s for r in ratios]
    vsize = sum(virtual_sizes)
    max_size = sum(sizes) * max_scale_up
    return int(vsize if vsize < max_size else max_size)


class CollateFormat(Enum):
    single = 1
    ordered_dict = 2


class SampledMultiDataset(FairseqDataset):
    """Samples from multiple sub-datasets according to given sampling ratios.
    Args:
        datasets (
            List[~torch.utils.data.Dataset]
            or OrderedDict[str, ~torch.utils.data.Dataset]
        ): datasets
        sampling_ratios (List[float]): list of probability of each dataset to be sampled
            (default: None, which corresponds to concating all dataset together).
        batch_by_size (bool): whether or not to batch by sequence length
            (default: True).
        seed (int): RNG seed to use (default: 2).
        epoch (int): starting epoch number (default: 1).
        eval_key (str, optional): a key used at evaluation time that causes
            this instance to pass-through batches from *datasets[eval_key]*.
        collate_format (CollateFormat):  collater output format, either CollateFormat.ordered_dict or
            CollateFormat.single (default: CollateFormat.single) where CollateFormat.single configures
            the collater to output batches of data mixed from all sub-datasets,
            and CollateFormat.ordered_dict configures the collater to output a dictionary of batches indexed by keys
            of sub-datasets.
            Note that not all sub-datasets will present in a single batch in both formats.
        virtual_size (int, or callable): the expected virtual size of the dataset (default: default_virtual_size_func).
        split (str): the split of the data, e.g. 'train', 'valid' or 'test'.
        shared_collater (bool): whether or not to all sub-datasets have the same collater.
    """

    def __init__(
            self,
            datasets,
            sampling_ratios=None,
            batch_by_size=False,
            seed=2,
            epoch=1,
            eval_key=None,
            collate_format=CollateFormat.single,
            virtual_size=default_virtual_size_func,
            split='',
            shared_collater=False,
    ):
        super().__init__()
        self.batch_by_size = batch_by_size
        self.shared_collater = shared_collater

        if isinstance(datasets, OrderedDict):
            self.keys = list(datasets.keys())
            datasets = list(datasets.values())
        elif isinstance(datasets, List):
            self.keys = list(range(len(datasets)))
        else:
            raise AssertionError()
        self.datasets = datasets
        self.split = split

        self.eval_key = eval_key
        if self.eval_key is not None:
            self.collate_format = CollateFormat.single
        else:
            self.collate_format = collate_format

        self.seed = seed
        self._cur_epoch = None
        self._cur_indices = None
        self._sizes = None
        self._ordered_indices = None
        self.virtual_size_per_dataset = None
        # caching properties
        self._reset_cached_properties()
        self.setup_sampling(sampling_ratios, virtual_size)
        self.cumulated_sizes = None
        self.virtual_size_per_dataset = None
        self._size_cache = {}
        self.set_epoch(epoch)

    def _reset_cached_properties(self):
        self._sizes = None
        self._ordered_indices = None
        self._cur_indices = None

    def setup_sampling(self, sample_ratios, virtual_size):
        sizes = [len(d) for d in self.datasets]
        if sample_ratios is None:
            # default back to concating datasets
            self.sample_ratios = None
            self.virtual_size = sum(sizes)
        else:
            if not isinstance(sample_ratios, np.ndarray):
                sample_ratios = np.array(sample_ratios)
            self.sample_ratios = plasma_utils.PlasmaArray(sample_ratios)
            virtual_size = default_virtual_size_func if virtual_size is None else virtual_size
            self.virtual_size = (
                virtual_size(self.datasets, self.sample_ratios.array) if callable(virtual_size)
                else virtual_size)

    def adjust_sampling(self, epoch, sampling_ratios, virtual_size):
        if sampling_ratios is not None:
            sampling_ratios = self._sync_sample_ratios(sampling_ratios)
            self.setup_sampling(sampling_ratios, virtual_size)

    def _sync_sample_ratios(self, ratios):
        # in case the ratios are not precisely the same across processes
        # also to ensure every procresses update the ratios in the same pace
        ratios = torch.DoubleTensor(ratios)
        if torch.distributed.is_initialized():
            if torch.cuda.is_available():
                distributed_utils.all_reduce(ratios.cuda())
            else:
                distributed_utils.all_reduce(ratios)
            ret = ratios.cpu()
            ret = ret.numpy()
        return ret

    def random_choice_in_dataset(self, rng, dataset, choice_size):
        if hasattr(dataset, 'random_choice_in_dataset'):
            return dataset.random_choice_in_dataset(rng, choice_size)
        dataset_size = len(dataset)
        return rng.choice(dataset_size, choice_size, replace=(choice_size > dataset_size))

    def get_virtual_indices(self, rng, datasets, sample_ratios, virtual_size):
        def get_counts(sample_ratios):
            counts = np.array([virtual_size * r for r in sample_ratios], dtype=np.int64)
            diff = virtual_size - counts.sum()
            assert diff >= 0
            # due to round-offs, the size might not match the desired sizes
            if diff > 0:
                dataset_indices = rng.choice(len(sample_ratios), size=diff, p=sample_ratios)
                for i in dataset_indices:
                    counts[i] += 1
            return counts

        def get_in_dataset_indices(datasets, sizes, sample_ratios):
            counts = get_counts(sample_ratios)
            # uniformally sample desired counts for each dataset
            # if the desired counts are large, sample with replacement:
            indices = [
                self.random_choice_in_dataset(rng, d, c)
                for c, d in zip(counts, datasets)]
            return indices

        sizes = [len(d) for d in datasets]
        if sample_ratios is None:
            # default back to concating datasets
            in_dataset_indices = [list(range(s)) for s in sizes]
            virtual_sizes_per_dataset = sizes
        else:
            sample_ratios = sample_ratios.array
            ratios = sample_ratios / sample_ratios.sum()
            in_dataset_indices = get_in_dataset_indices(datasets, sizes, ratios)
            virtual_sizes_per_dataset = [len(d) for d in in_dataset_indices]
        virtual_sizes_per_dataset = np.array(virtual_sizes_per_dataset, np.int64)
        cumulative_sizes = np.cumsum(virtual_sizes_per_dataset)
        assert sum(virtual_sizes_per_dataset) == virtual_size
        assert cumulative_sizes[-1] == virtual_size
        if virtual_size < sum(sizes):
            logger.warning(
                f'virtual data size ({virtual_size}) is less than real data size ({sum(sizes)}).'
                ' If virtual size << real data size, there could be data coverage issue.'
            )
        in_dataset_indices = np.hstack(in_dataset_indices)
        return in_dataset_indices, cumulative_sizes, virtual_sizes_per_dataset

    def _get_dataset_and_index(self, index):
        i = bisect_right(self.cumulated_sizes.array, index)
        return i, self._cur_indices.array[index]

    def __getitem__(self, index):
        ds_idx, ds_sample_idx = self._get_dataset_and_index(index)
        return (ds_idx, self.datasets[ds_idx][ds_sample_idx])

    def num_tokens(self, index):
        ds_idx, ds_sample_idx = self._get_dataset_and_index(index)
        return self.datasets[ds_idx].num_tokens(ds_sample_idx)

    def size(self, index):
        if self._sizes is not None:
            return self._sizes[index]
        ds_idx, ds_sample_idx = self._get_dataset_and_index(index)
        return self.datasets[ds_idx].size(ds_sample_idx)

    def __len__(self):
        return self.virtual_size

    def collater(self, samples, **extra_args):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return None
        if self.collate_format == 'ordered_dict':
            collect_samples = [[] for _ in range(len(self.datasets))]
            for (i, sample) in samples:
                collect_samples[i].append(sample)
            return OrderedDict([
                (self.keys[i], dataset.collater(collect_samples[i]))
                for i, (key, dataset) in enumerate(zip(self.keys, self.datasets))
                if len(collect_samples[i]) > 0
            ])
        elif self.shared_collater:
            return self.datasets[0].collater(
                [s for _, s in samples]
            )
        else:
            samples_dict = defaultdict(list)
            max_size = defaultdict(int)
            for ds_idx, s in samples:
                max_size['source'] = max(max_size['source'], s['source'].size(0))
                if s['target'] is not None:
                    max_size['target'] = max(max_size['target'], s['target'].size(0))
                samples_dict[ds_idx].append(s)
            batches = [
                self.datasets[i].collater(samples_dict[i], max_size=max_size)
                for i in range(len(self.datasets))
                if len(samples_dict[i]) > 0
            ]

            def straight_data(tensors):
                batch = torch.cat(tensors, dim=0)
                return batch

            src_lengths = straight_data([b['net_input']['src_lengths'] for b in batches])
            src_lengths, sort_order = src_lengths.sort(descending=True)

            def straight_order(tensors):
                batch = straight_data(tensors)
                return batch.index_select(0, sort_order)

            batch = {
                'id': straight_order([b['id'] for b in batches]),
                'nsentences': sum(b['nsentences'] for b in batches),
                'ntokens': sum(b['ntokens'] for b in batches),
                'net_input': {
                    'src_tokens': straight_order([b['net_input']['src_tokens'] for b in batches]),
                    'src_lengths': src_lengths,
                },
                'target': straight_order([b['target'] for b in batches]) if batches[0]['target'] is not None else None,
            }
            if 'prev_output_tokens' in batches[0]['net_input']:
                batch['net_input']['prev_output_tokens'] = straight_order(
                    [b['net_input']['prev_output_tokens'] for b in batches])
            if 'src_lang_id' in batches[0]['net_input']:
                batch['net_input']['src_lang_id'] = straight_order([b['net_input']['src_lang_id'] for b in batches])
            if 'tgt_lang_id' in batches[0]:
                batch['tgt_lang_id'] = straight_order([b['tgt_lang_id'] for b in batches])
            return batch

    @property
    def sizes(self):
        if self._sizes is not None:
            return self._sizes
        start_time = time.time()
        size_cache = self._size_cache
        ret = []
        for i in range(len(self)):
            ds_idx, ds_sample_idx = self._get_dataset_and_index(i)
            if (ds_idx, ds_sample_idx) in size_cache:
                ret.append(size_cache[(ds_idx, ds_sample_idx)])
            else:
                s = self.datasets[ds_idx].size(ds_sample_idx)
                size_cache[(ds_idx, ds_sample_idx)] = s
                ret.append(s)
        logger.debug(f'sizes() calling time: {get_time_gap(start_time, time.time())}')
        self._sizes = np.array(ret, np.int64)
        return self._sizes

    def ordered_indices(self):
        if self._ordered_indices is not None:
            return self._ordered_indices

        if self.batch_by_size:
            # No need to do shuffle as the data items are already randomized
            indices = np.arange(len(self))
            sizes = self.sizes
            tgt_sizes = sizes[:, 1] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else None
            src_sizes = sizes[:, 0] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else sizes

            # sort by target length, then source length
            if tgt_sizes is not None:
                indices = indices[
                    np.argsort(tgt_sizes[indices], kind='mergesort')
                ]
            sort_indices = indices[np.argsort(src_sizes[indices], kind='mergesort')]
        else:
            sort_indices = np.arange(len(self))
        self._ordered_indices = sort_indices
        return sort_indices

    def prefetch(self, indices):
        prefetch_indices = [[] for _ in range(len(self.datasets))]
        for i in indices:
            ds_idx, ds_sample_idx = self._get_dataset_and_index(i)
            prefetch_indices[ds_idx].append(ds_sample_idx)
        for i in range(len(prefetch_indices)):
            self.datasets[i].prefetch(prefetch_indices[i])

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if epoch == self._cur_epoch:
            # re-enter so return
            return
        for d in self.datasets:
            if hasattr(d, 'set_epoch'):
                d.set_epoch(epoch)
        self._cur_epoch = epoch
        self._establish_virtual_datasets()

    def _establish_virtual_datasets(self):
        if self.sample_ratios is None and self._cur_indices is not None:
            # not a samping dataset, no need to resample if indices are already established
            return
        self._reset_cached_properties()

        start_time = time.time()
        # Generate a weighted sample of indices as a function of the
        # random seed and the current epoch.
        rng = np.random.RandomState(
           [
               int(hashlib.sha1(str(self.__class__.__name__).encode('utf-8')).hexdigest(), 16) % (2 ** 32),
               self.seed % (2 ** 32),  # global seed
               self._cur_epoch,  # epoch index,
           ]
        )
        indices, cumulated_sizes, virtual_size_per_dataset = self.get_virtual_indices(
            rng, self.datasets, self.sample_ratios, self.virtual_size)
        self._cur_indices = plasma_utils.PlasmaArray(indices)
        self.cumulated_sizes = plasma_utils.PlasmaArray(cumulated_sizes)
        self.virtual_size_per_dataset = plasma_utils.PlasmaArray(virtual_size_per_dataset)

        logger.info(f'[{self.split}] Raw sizes: {str(dict(zip(self.keys, [len(d) for d in self.datasets])))}')
        logger.info(f'[{self.split}] Resampled sizes: {str(dict(zip(self.keys, self.virtual_size_per_dataset.array)))}')
        if self.sample_ratios is not None:
            logger.info(f'[{self.split}] Upsampling ratios: {str(dict(zip(self.keys, self.sample_ratios.array)))}')
        else:
            logger.info(f'[{self.split}] A concat dataset')
        logger.debug(f'[{self.split}] virtual dataset established time: {get_time_gap(start_time, time.time())}')
