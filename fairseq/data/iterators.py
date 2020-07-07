# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import math
import operator
import os
import queue
import time
from threading import Thread

import numpy as np
import torch

from fairseq.data import data_utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Object used by _background_consumer to signal the source is exhausted
# to the main thread.
_sentinel = object()


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by
            ``__len__``. This can be used to truncate *iterator*.

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=None, total=None):
        self.iterable = iterable
        self.itr = iter(self)

        if start is None:
            self.n = getattr(iterable, 'n', 0)
        else:
            self.n = start

        if total is None:
            self.total = self.n + len(iterable)
        else:
            self.total = total

    def __len__(self):
        return self.total

    def __iter__(self):
        for x in self.iterable:
            if self.n >= self.total:
                return
            self.n += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.n < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self

    def take(self, n):
        """
        Truncates the iterator to n elements at most.
        """
        self.total = min(self.total, n)

        # Propagate this change to the underlying iterator
        if hasattr(self.iterable, "take"):
            self.iterable.take(n)


class EpochBatchIterating(object):
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def next_epoch_idx(self):
        raise NotImplementedError

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        raise NotImplementedError

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        raise NotImplementedError

    @property
    def iterations_in_epoch(self) -> int:
        """The number of consumed batches in the current epoch."""
        raise NotImplementedError

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        raise NotImplementedError


class StreamingEpochBatchIterator(EpochBatchIterating):
    def __init__(
        self, dataset, epoch=1, num_shards=1, shard_id=0,
    ):
        assert isinstance(dataset, torch.utils.data.IterableDataset)
        self.dataset = dataset
        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self._current_epoch_iterator = None
        self.num_shards = num_shards
        self.shard_id = shard_id

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._current_epoch_iterator is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        self.epoch = self.next_epoch_idx
        self.dataset.set_epoch(self.epoch)
        self._current_epoch_iterator = CountingIterator(
            iterable=ShardedIterator(
                iterable=self.dataset,
                num_shards=self.num_shards,
                shard_id=self.shard_id,
            ),
        )
        return self._current_epoch_iterator

    def end_of_epoch(self) -> bool:
        return not self._current_epoch_iterator.has_next()

    @property
    def iterations_in_epoch(self) -> int:
        if self._current_epoch_iterator is not None:
            return self._current_epoch_iterator.n
        return 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']


class EpochBatchIterator(EpochBatchIterating):
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
            indices
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).
        buffer_size (int, optional): the number of batches to keep ready in the
            queue. Helps speeding up dataloading. When buffer_size is zero, the
            default torch.utils.data.DataLoader preloading is used.
        timeout (int, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
    """

    def __init__(
        self, dataset, collate_fn, batch_sampler, seed=1, num_shards=1, shard_id=0,
        num_workers=0, epoch=1, buffer_size=0, timeout=0,
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.frozen_batches = tuple(batch_sampler)
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers
        # This upper limit here is to prevent people from abusing this feature
        # in a shared computing environment.
        self.buffer_size = min(buffer_size, 20)
        self.timeout = timeout

        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self.shuffle = True
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset, 'supports_prefetch', False)

    def __len__(self):
        return int(math.ceil(len(self.frozen_batches) / float(self.num_shards)))

    @property
    def n(self):
        return self.iterations_in_epoch

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._next_epoch_itr is not None:
            return self.epoch
        elif self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        self.epoch = self.next_epoch_idx
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus,
            )
        self.dataset.set_epoch(self.epoch)
        self.shuffle = shuffle
        return self._cur_epoch_itr

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.n
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.n
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
            'shuffle': self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle=state_dict.get('shuffle', True),
                offset=itr_pos,
            )
            if self._next_epoch_itr is None:
                # we finished the epoch, increment epoch counter
                self.epoch += 1
        else:
            self._next_epoch_itr = None

    def _get_iterator_for_epoch(self, epoch, shuffle, fix_batches_to_gpus=False, offset=0):

        def shuffle_batches(batches, seed):
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
            return batches

        if self._supports_prefetch:
            batches = self.frozen_batches

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)

            batches = list(ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            else:
                batches = self.frozen_batches
            batches = list(ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
            timeout=self.timeout,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CoutingIterator
        itr = CountingIterator(itr, start=offset)
        return itr


class GroupedIterator(CountingIterator):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, chunk_size):
        itr = _chunk_iterator(iterable, chunk_size)
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, 'n', 0) / float(chunk_size))),
            total=int(math.ceil(len(iterable) / float(chunk_size))),
        )
        self.chunk_size = chunk_size


def _chunk_iterator(itr, chunk_size):
    chunk = []
    for x in itr:
        chunk.append(x)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class ShardedIterator(CountingIterator):
    """A sharded wrapper around an iterable, padded to length.

    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')
        sharded_len = int(math.ceil(len(iterable) / float(num_shards)))
        itr = map(
            operator.itemgetter(1),
            itertools.zip_longest(
                range(sharded_len),
                itertools.islice(iterable, shard_id, len(iterable), num_shards),
                fillvalue=fill_value,
            ),
        )
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, 'n', 0) / float(num_shards))),
            total=sharded_len,
        )


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len):
        Thread.__init__(self)

        self._queue = queue
        self._source = source
        self._max_len = max_len
        self.count = 0

    def run(self):
        try:
            self._source_iter = iter(self._source)
            for _ in range(len(self._source)):
                item = next(self._source_iter)
                self._queue.put(item)

                # Stop if we reached the maximum length
                self.count += 1
                if self._max_len is not None and self.count >= self._max_len:
                    break

            # Signal the consumer we are done.
            self._queue.put(_sentinel)
        except Exception as e:
            self._queue.put(e)

        del self._source_iter


class BufferedIterator(object):
    def __init__(self, size, iterable):
        self._queue = queue.Queue(size)
        self._iterable = iterable
        self.max_len = None
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(
            self._queue,
            self._iterable,
            self.max_len
        )
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iterable)

    def take(self, n):
        self.max_len = n

    def __next__(self):
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()

        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < max(1, self._queue.maxsize // 2):
            if time.time() - self.start_time > 5 * 60:
                if self.warning_time is None or time.time() - self.warning_time > 15 * 60:
                    logger.info(
                        "Data loading buffer is empty or nearly empty. This may "
                        "indicate a data loading bottleneck, and increasing the "
                        "number of workers (--num-workers) may help."
                    )
                    self.warning_time = time.time()

        # Get next example
        item = self._queue.get(True)
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            raise StopIteration()
        return item
