# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import math
import logging
import time

import numpy as np
import torch
from fairseq import distributed_utils
from fairseq.data import plasma_utils, SampledMultiDataset
from .sampled_multi_dataset import default_virtual_size_func, get_time_gap, CollateFormat


logger = logging.getLogger(__name__)


class SampledMultiEpochDataset(SampledMultiDataset):
    """Samples from multiple sub-datasets according to sampling ratios
       using virtual epoch sizes to speed up dataloading.
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
        virtual_epoch_size (int): virtual epoch size, the dataset will go through the data by
            this virtual epoch size one by one to speed up data loading, e.g. indicing and filtering
            can be performed whenever a virtual epoch is loaded without waiting for the whole dataset to be loaded.
        shared_collater (bool): whether or not to all sub-datasets have the same collater.
        shard_epoch (int): the real epoch number for shard selection.
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
        virtual_epoch_size=None,
        shared_collater=False,
        shard_epoch=1,
    ):
        self.virtual_epoch_size = virtual_epoch_size
        self._current_epoch_start_index = None
        self._epoch_sizes = None
        self._epoch_ordered_indices = None
        self._random_globa_indices = None
        self.shard_epoch = shard_epoch if shard_epoch is not None else 1
        self.load_next_shard = None
        super().__init__(
            datasets=datasets,
            sampling_ratios=sampling_ratios,
            batch_by_size=batch_by_size,
            seed=seed,
            epoch=epoch,
            eval_key=eval_key,
            collate_format=collate_format,
            virtual_size=virtual_size,
            split=split,
            shared_collater=shared_collater,
        )

    def _setup(self, epoch):
        self.virtual_epoch_size = self.virtual_epoch_size if self.virtual_epoch_size is not None else self.virtual_size
        if self.virtual_epoch_size > self.virtual_size:
            logger.warning(f'virtual epoch size {self.virtual_epoch_size} '
                           f'is greater than virtual dataset size {self.virtual_size}')
            self.virtual_epoch_size = self.virtual_size
        self.num_virtual_epochs = math.ceil(self.virtual_size / self.virtual_epoch_size)
        self._current_epoch_start_index = self._get_epoch_start_index(epoch)
        logger.info(f'virtual epoch size {self.virtual_epoch_size}; virtual dataset size {self.virtual_size}')

    def _map_epoch_index_to_global(self, index):
        index = self._current_epoch_start_index + index
        # add randomness
        return self._random_globa_indices.array[index]

    def __getitem__(self, index):
        i = self._map_epoch_index_to_global(index)
        return super().__getitem__(i)

    def num_tokens(self, index):
        i = self._map_epoch_index_to_global(index)
        return super().num_tokens(i)

    def size(self, index):
        if self._epoch_sizes is not None:
            return self._epoch_sizes.array[index]
        index = self._map_epoch_index_to_global(index)
        ds_idx, ds_sample_idx = self._get_dataset_and_index(index)
        return self.datasets[ds_idx].size(ds_sample_idx)

    def __len__(self):
        return (
            self.virtual_epoch_size
            if self._current_epoch_start_index + self.virtual_epoch_size < self.virtual_size
            else self.virtual_size - self._current_epoch_start_index
        )

    @property
    def sizes(self):
        if self._epoch_sizes is not None:
            return self._epoch_sizes.array
        start_time = time.time()

        size_cache = self._size_cache
        ret = []
        for i in range(len(self)):
            index = self._map_epoch_index_to_global(i)
            ds_idx, ds_sample_idx = self._get_dataset_and_index(index)

            if (ds_idx, ds_sample_idx) in size_cache:
                ret.append(size_cache[(ds_idx, ds_sample_idx)])
            else:
                s = self.datasets[ds_idx].size(ds_sample_idx)
                s = (s, s) if not isinstance(s, tuple) else s
                size_cache[(ds_idx, ds_sample_idx)] = s
                ret.append(s)
        self._epoch_sizes = plasma_utils.PlasmaArray(np.array(ret, np.int64))
        logger.info(f'sizes() calling time: {get_time_gap(start_time, time.time())}')
        return self._epoch_sizes.array

    def ordered_indices(self):
        if self._epoch_ordered_indices is not None:
            return self._epoch_ordered_indices.array

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
        self._epoch_ordered_indices = plasma_utils.PlasmaArray(sort_indices)
        return self._epoch_ordered_indices.array

    def prefetch(self, indices):
        prefetch_indices = [[] for _ in range(len(self.datasets))]
        for i in indices:
            index = self._map_epoch_index_to_global(i)
            ds_idx, ds_sample_idx = self._get_dataset_and_index(index)
            prefetch_indices[ds_idx].append(ds_sample_idx)
        for i in range(len(prefetch_indices)):
            self.datasets[i].prefetch(prefetch_indices[i])

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch):
        if self._current_epoch_start_index is None:
            self._setup(epoch)
            self._next_virtual_epoch(epoch)
        if epoch == self._cur_epoch:
            # re-enter so return
            return
        self._next_virtual_epoch(epoch)

    def _get_epoch_start_index(self, epoch):
        assert epoch >= 1  # fairseq is using 1-based epoch everywhere
        return ((epoch - 1) % self.num_virtual_epochs) * self.virtual_epoch_size

    def _next_global_indices(self, epoch):
        rng = np.random.RandomState(
           [
               int(hashlib.sha1(str(self.__class__.__name__).encode('utf-8')).hexdigest(), 16) % (2 ** 32),
               self.seed % (2 ** 32),  # global seed
               epoch,  # epoch index,
           ]
        )
        del self._random_globa_indices
        self._random_globa_indices = plasma_utils.PlasmaArray(
            rng.choice(self.virtual_size, self.virtual_size, replace=False))
        if self.load_next_shard is None:
            self.load_next_shard = False
        else:
            # increase shard epoch for next loading
            self.shard_epoch += 1
            self.load_next_shard = True
            # a hack to avoid possible out of sync of shard epoch number
            # TODO: to confirm whether this is needed; without it, CUDA event error is occassionally observed
            synced_shard_epoch = self._sync_shard_epoch(self.shard_epoch)
            logger.info('to load next epoch/shard in next load_dataset: '
                        f'epoch={epoch}/shard_epoch={self.shard_epoch}[synced={synced_shard_epoch}]')

    def _sync_shard_epoch(self, shard_epoch):
        # in case the ratios are not precisely the same across processes
        # also to ensure every procresses update the ratios in the same pace
        shard_epoch = torch.DoubleTensor([shard_epoch])
        if torch.distributed.is_initialized():
            if torch.cuda.is_available():
                distributed_utils.all_reduce(shard_epoch.cuda())
            else:
                distributed_utils.all_reduce(shard_epoch)
            ret = shard_epoch.cpu()
            ret = ret.numpy()
        return ret

    def _sync_epoch(self, epoch):
        # in case the ratios are not precisely the same across processes
        # also to ensure every procresses update the ratios in the same pace
        epoch = torch.DoubleTensor([epoch])
        if torch.distributed.is_initialized():
            if torch.cuda.is_available():
                distributed_utils.all_reduce(epoch.cuda())
            else:
                distributed_utils.all_reduce(epoch)
            ret = epoch.cpu()
            ret = ret.numpy()
        return ret

    def _next_virtual_epoch(self, epoch):
        index = self._get_epoch_start_index(epoch)
        if index == 0 or self._random_globa_indices is None:
            # need to start from the beginning,
            # so call super().set_epoch(epoch) to establish the global virtual indices
            logger.info('establishing a new set of global virtual indices for '
                        f'epoch={epoch}/shard_epoch={self.shard_epoch}')
            super().set_epoch(epoch)
            self._next_global_indices(epoch)
        else:
            self._cur_epoch = epoch
        # reset cache sizes and ordered_indices for the epoch after moving to a new epoch

        self._clean_if_not_none([
            self._epoch_sizes, self._epoch_ordered_indices, self._size_cache
        ])
        self._epoch_sizes = None
        self._epoch_ordered_indices = None
        self._current_epoch_start_index = index
        self._size_cache = {}
