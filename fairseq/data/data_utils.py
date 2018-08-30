# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import itertools
import os

import numpy as np
import torch


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


class ShardedIterator(object):
    """A sharded wrapper around an iterable (padded to length)."""

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value,
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count."""

    def __init__(self, iterable):
        self.iterable = iterable
        self.count = 0
        self.itr = iter(self)

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        for x in self.iterable:
            self.count += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        return self.count < len(self)

    def skip(self, num_to_skip):
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class EpochBatchIterator(object):
    """A multi-epoch iterator over a :class:`~torch.utils.data.Dataset`.

    Compared to :class:`~torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the ``num_shards`` and ``shard_id`` arguments

    Args:
        dataset (Dataset): dataset from which to load the data
        batch_sampler (Sampler): an iterator over batches of indices
        seed (int, optional): seed for random number generator for
            reproducibility. Default: ``1``
        num_shards (int, optional): shard the data iterator into N
            shards. Default: ``1``
        shard_id (int, optional): which shard of the data iterator to
            return. Default: ``0``
    """

    def __init__(self, dataset, batch_sampler, seed=1, num_shards=1, shard_id=0):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.frozen_batches = tuple(batch_sampler)
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.epoch = 0
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

    def __len__(self):
        return len(self.frozen_batches)

    def next_epoch_itr(self, shuffle=True):
        """
        Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator. Default: ``True``
        """
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(self.epoch, shuffle)
        return self._cur_epoch_itr

    def end_of_epoch(self):
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            itr = self._get_iterator_for_epoch(self.epoch, state_dict.get('shuffle', True))
            if itr_pos < len(itr):
                self._next_epoch_itr = itr.skip(itr_pos)

    def _get_iterator_for_epoch(self, epoch, shuffle):
        if shuffle:
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with numpy_seed(self.seed + epoch):
                batches = list(self.frozen_batches)  # copy
                np.random.shuffle(batches)
        else:
            batches = self.frozen_batches
        return CountingIterator(torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collater,
            batch_sampler=ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[]),
        ))


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def filter_by_size(indices, size_fn, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        size_fn (callable): function that returns the size of a given index
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception
            if any elements are filtered. Default: ``False``
    """
    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) < max_positions
        else:
            return all(a <= b for a, b in zip(size_fn(idx), max_positions))

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    for idx in itr:
        if len(ignored) > 0 and raise_exception:
            raise Exception((
                'Size of sample #{} is invalid (={}) since max_positions={}, '
                'skip this example with --skip-invalid-size-inputs-valid-test'
            ).format(idx, self.size(idx), max_positions))
        yield idx

    if len(ignored) > 0:
        print((
            '| WARNING: {} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch.
            Default: ``None``
        max_sentences (int, optional): max number of sentences in each
            batch. Default: ``None``
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N. Default: ``1``
    """
    max_tokens = max_tokens if max_tokens is not None else float('Inf')
    max_sentences = max_sentences if max_sentences is not None else float('Inf')
    bsz_mult = required_batch_size_multiple

    batch = []

    def is_batch_full(num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        return False

    sample_len = 0
    sample_lens = []
    ignored = []
    for idx in indices:
        sample_lens.append(num_tokens_fn(idx))
        sample_len = max(sample_len, sample_lens[-1])
        num_tokens = (len(batch) + 1) * sample_len
        if is_batch_full(num_tokens):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            yield batch[:mod_len]
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

        batch.append(idx)

    if len(batch) > 0:
        yield batch
