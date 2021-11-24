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

# Object used by _background_consumer to signal the source is exhausted
# to the main thread.
_sentinel = object()


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by ``__len``.
            This can be used to truncate *iterator*.

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=None, total=None):
        self._itr = iter(iterable)
        self.n = start or getattr(iterable, "n", 0)
        self.total = total if total is not None else self.n + len(iterable)

    def __len__(self):
        return self.total

    def __iter__(self):
        return self

    def __next__(self):
        if not self.has_next():
            raise StopIteration
        try:
            x = next(self._itr)
        except StopIteration:
            raise IndexError(
                f"Iterator expected to have length {self.total}, "
                "but exhausted at position {self.n}."
            )
        self.n += 1
        return x

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.n < self.total

    def skip(self, n):
        """Fast-forward the iterator by skipping n elements."""
        for _ in range(n):
            next(self)
        return self

    def take(self, n):
        """Truncate the iterator to n elements at most."""
        self.total = min(self.total, n)
        # Propagate this change to the underlying iterator
        if hasattr(self._itr, "take"):
            self._itr.take(max(n - self.n, 0))
        return self


class EpochBatchIterating(object):
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def next_epoch_idx(self):
        raise NotImplementedError

    def next_epoch_itr(
        self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True
    ):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus (bool, optional): ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
            set_dataset_epoch (bool, optional): update the wrapped Dataset with
                the new epoch number (default: True).
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

    @property
    def first_batch(self):
        return "DUMMY"


class StreamingEpochBatchIterator(EpochBatchIterating):
    """A steaming-style iterator over a :class:`torch.utils.data.IterableDataset`.

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        max_sentences: batch size
        collate_fn (callable): merges a list of samples to form a mini-batch
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).
        buffer_size (int, optional): the number of batches to keep ready in the
            queue. Helps speeding up dataloading. When buffer_size is zero, the
            default torch.utils.data.DataLoader preloading is used.
        timeout (int, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative (default: ``0``).
    """

    def __init__(
        self,
        dataset,
        max_sentences=1,
        collate_fn=None,
        epoch=1,
        num_workers=0,
        buffer_size=0,
        timeout=0,
    ):
        assert isinstance(dataset, torch.utils.data.IterableDataset)
        self.dataset = dataset
        self.max_sentences = max_sentences
        self.collate_fn = collate_fn
        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self.num_workers = num_workers
        # This upper limit here is to prevent people from abusing this feature
        # in a shared computing environment.
        self.buffer_size = min(buffer_size, 20)
        self.timeout = timeout

        self._current_epoch_iterator = None

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._current_epoch_iterator is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(
        self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True
    ):
        self.epoch = self.next_epoch_idx
        if set_dataset_epoch and hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        self._current_epoch_iterator = self._get_iterator_for_epoch(self.epoch, shuffle)
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
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]

    def _get_iterator_for_epoch(self, epoch, shuffle, offset=0):
        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        # Create data loader
        worker_init_fn = getattr(self.dataset, "worker_init_fn", None)
        itr = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.max_sentences,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            timeout=self.timeout,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CountingIterator
        itr = CountingIterator(itr, start=offset)

        return itr


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
        batch_sampler (~torch.utils.data.Sampler or a callable): an iterator over batches of
            indices, or a callable to create such an iterator (~torch.utils.data.Sampler).
            A callable batch_sampler will be called for each epoch to enable per epoch dynamic
            batch iterators defined by this callable batch_sampler.
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
            from workers. Should always be non-negative (default: ``0``).
        disable_shuffling (bool, optional): force disable shuffling
            (default: ``False``).
        skip_remainder_batch (bool, optional): if set, discard the last batch in an epoch
            for the sake of training stability, as the last batch is usually smaller than
                local_batch_size * distributed_word_size (default: ``False``).
        grouped_shuffling (bool, optional): enable shuffling batches in groups
            of num_shards. Ensures that each GPU receives similar length sequences when
            batches are sorted by length.
    """

    def __init__(
        self,
        dataset,
        collate_fn,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        buffer_size=0,
        timeout=0,
        disable_shuffling=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_sampler = batch_sampler
        self._frozen_batches = (
            tuple(batch_sampler) if not callable(batch_sampler) else None
        )
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers
        # This upper limit here is to prevent people from abusing this feature
        # in a shared computing environment.
        self.buffer_size = min(buffer_size, 20)
        self.timeout = timeout
        self.disable_shuffling = disable_shuffling
        self.skip_remainder_batch = skip_remainder_batch
        self.grouped_shuffling = grouped_shuffling

        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self.shuffle = not disable_shuffling
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset, "supports_prefetch", False)

    @property
    def frozen_batches(self):
        if self._frozen_batches is None:
            self._frozen_batches = tuple(self.batch_sampler(self.dataset, self.epoch))
        return self._frozen_batches

    @property
    def first_batch(self):
        if len(self.frozen_batches) == 0:
            raise Exception(
                "The dataset is empty. This could indicate "
                "that all elements in the dataset have been skipped. "
                "Try increasing the max number of allowed tokens or using "
                "a larger dataset."
            )

        if getattr(self.dataset, "supports_fetch_outside_dataloader", True):
            return self.collate_fn([self.dataset[i] for i in self.frozen_batches[0]])
        else:
            return "DUMMY"

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

    def next_epoch_itr(
        self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True
    ):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus (bool, optional): ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
            set_dataset_epoch (bool, optional): update the wrapped Dataset with
                the new epoch number (default: True).
        """
        if self.disable_shuffling:
            shuffle = False
        prev_epoch = self.epoch
        self.epoch = self.next_epoch_idx
        if set_dataset_epoch and hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            if callable(self.batch_sampler) and prev_epoch != self.epoch:
                # reset _frozen_batches to refresh the next epoch
                self._frozen_batches = None
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus,
            )
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
        if self.end_of_epoch():
            epoch = self.epoch + 1
            iter_in_epoch = 0
        else:
            epoch = self.epoch
            iter_in_epoch = self.iterations_in_epoch
        return {
            "version": 2,
            "epoch": epoch,
            "iterations_in_epoch": iter_in_epoch,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict["epoch"]
        itr_pos = state_dict.get("iterations_in_epoch", 0)
        version = state_dict.get("version", 1)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle=state_dict.get("shuffle", True), offset=itr_pos,
            )
            if self._next_epoch_itr is None:
                if version == 1:
                    # legacy behavior: we finished the epoch, increment epoch counter
                    self.epoch += 1
                else:
                    raise RuntimeError(
                        "Cannot resume training due to dataloader mismatch, please "
                        "report this to the fairseq developers. You can relaunch "
                        "training with `--reset-dataloader` and it should work."
                    )
        else:
            self._next_epoch_itr = None

    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        def shuffle_batches(batches, seed):
            with data_utils.numpy_seed(seed):

                if self.grouped_shuffling:
                    grouped_batches = [
                        batches[(i * self.num_shards) : ((i + 1) * self.num_shards)]
                        for i in range((len(batches) // self.num_shards))
                    ]
                    np.random.shuffle(grouped_batches)
                    batches = list(itertools.chain(*grouped_batches))
                else:
                    np.random.shuffle(batches)

            return batches

        if self._supports_prefetch:
            batches = self.frozen_batches

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)

            batches = list(
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            else:
                batches = self.frozen_batches
            batches = list(
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
            timeout=self.timeout,
            pin_memory=True,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CountingIterator
        itr = CountingIterator(itr, start=offset)

        if self.skip_remainder_batch:
            # TODO: Below is a lazy implementation which discard the final batch regardless
            # of whether it is a full batch or not.
            total_num_itrs = len(batches) - 1
            itr.take(total_num_itrs)
            logger.info(f"skip final residual batch, total_num_itrs = {total_num_itrs}")

        return itr


class GroupedIterator(CountingIterator):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
        skip_remainder_batch (bool, optional): if set, discard the last grouped batch in
          each training epoch, as the last grouped batch is usually smaller than
                local_batch_size * distributed_word_size * chunk_size (default: ``False``).
    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, chunk_size, skip_remainder_batch=False):
        if skip_remainder_batch:
            total_num_itrs = int(math.floor(len(iterable) / float(chunk_size)))
            logger.info(
                f"skip final residual batch, grouped total_num_itrs = {total_num_itrs}"
            )
        else:
            total_num_itrs = int(math.ceil(len(iterable) / float(chunk_size)))
            logger.info(f"grouped total_num_itrs = {total_num_itrs}")

        itr = _chunk_iterator(iterable, chunk_size, skip_remainder_batch)
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, "n", 0) / float(chunk_size))),
            total=total_num_itrs,
        )
        self.chunk_size = chunk_size

        if skip_remainder_batch:
            self.take(total_num_itrs)
            # TODO: [Hack] Here the grouped iterator modifies the base iterator size so that
            # training can move into the next epoch once the grouped iterator is exhausted.
            # Double-check this implementation in case unexpected behavior occurs.
            iterable.take(total_num_itrs * chunk_size)


def _chunk_iterator(itr, chunk_size, skip_remainder_batch=False):
    chunk = []
    for x in itr:
        chunk.append(x)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if not skip_remainder_batch and len(chunk) > 0:
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

    def __init__(
        self, iterable, num_shards, shard_id, fill_value=None, skip_remainder_batch=None
    ):
        """
        Args:
            skip_remainder_batch: ignored"""
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError("shard_id must be between 0 and num_shards")
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
            start=int(math.ceil(getattr(iterable, "n", 0) / float(num_shards))),
            total=sharded_len,
        )


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len, cuda_device):
        Thread.__init__(self)

        self._queue = queue
        self._source = source
        self._max_len = max_len
        self.count = 0
        self.cuda_device = cuda_device

    def run(self):
        # set_device to avoid creation of GPU0 context when using pin_memory
        if self.cuda_device is not None:
            torch.cuda.set_device(self.cuda_device)

        try:
            for item in self._source:
                self._queue.put(item)

                # Stop if we reached the maximum length
                self.count += 1
                if self._max_len is not None and self.count >= self._max_len:
                    break

            # Signal the consumer we are done.
            self._queue.put(_sentinel)
        except Exception as e:
            self._queue.put(e)


class BufferedIterator(object):
    def __init__(self, size, iterable):
        self._queue = queue.Queue(size)
        self._iterable = iterable
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None

        self.total = len(iterable)

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(
            self._queue,
            self._iterable,
            self.total,
            torch.cuda.current_device() if torch.cuda.is_available() else None,
        )
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def take(self, n):
        self.total = min(self.total, n)
        # Propagate this change to the underlying iterator
        if hasattr(self._iterable, "take"):
            self._iterable.take(n)
        return self

    def __next__(self):
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()

        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < min(2, max(1, self._queue.maxsize // 2)):
            if time.time() - self.start_time > 5 * 60:
                if (
                    self.warning_time is None
                    or time.time() - self.warning_time > 15 * 60
                ):
                    logger.debug(
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


class GroupedEpochBatchIterator(EpochBatchIterator):
    """Grouped version of EpochBatchIterator
    It takes several samplers from different datasets.
    Each epoch shuffle the dataset wise sampler individually with different
    random seed. The those sub samplers are combined with into
    one big samplers with deterministic permutation to mix batches from
    different datasets. It will act like EpochBatchIterator but make sure
    1) data from one data set each time
    2) for different workers, they use the same order to fetch the data
    so they will use data from the same dataset everytime
    mult_rate is used for update_freq > 1 case where we want to make sure update_freq
    mini-batches come from same source
    """

    def __init__(
        self,
        dataset,
        collate_fn,
        batch_samplers,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
        mult_rate=1,
        buffer_size=0,
    ):
        super().__init__(
            dataset,
            collate_fn,
            batch_samplers,
            seed,
            num_shards,
            shard_id,
            num_workers,
            epoch,
            buffer_size,
        )
        # level 0: sub-samplers 1: batch_idx 2: batches
        self._frozen_batches = tuple([tuple(sub_batch) for sub_batch in batch_samplers])
        self.step_size = mult_rate * num_shards

        self.lengths = [
            (len(x) // self.step_size) * self.step_size for x in self.frozen_batches
        ]

    def __len__(self):
        return sum(self.lengths)

    @property
    def first_batch(self):
        if len(self.frozen_batches) == 0:
            raise Exception(
                "The dataset is empty. This could indicate "
                "that all elements in the dataset have been skipped. "
                "Try increasing the max number of allowed tokens or using "
                "a larger dataset."
            )

        if self.dataset.supports_fetch_outside_dataloader:
            return self.collate_fn([self.dataset[i] for i in self.frozen_batches[0][0]])
        else:
            return "DUMMY"

    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        def shuffle_batches(batches, seed):
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
            return batches

        def return_full_batches(batch_sets, seed, shuffle):
            if shuffle:
                batch_sets = [shuffle_batches(list(x), seed) for x in batch_sets]

            batch_sets = [
                batch_sets[i][: self.lengths[i]] for i in range(len(batch_sets))
            ]
            batches = list(itertools.chain.from_iterable(batch_sets))

            if shuffle:
                with data_utils.numpy_seed(seed):
                    idx = np.random.permutation(len(batches) // self.step_size)
                    if len(idx) * self.step_size != len(batches):
                        raise ValueError(
                            "ERROR: %d %d %d %d"
                            % (len(idx), self.step_size, len(batches), self.shard_id),
                            ":".join(["%d" % x for x in self.lengths]),
                        )
                    mini_shards = [
                        batches[i * self.step_size : (i + 1) * self.step_size]
                        for i in idx
                    ]
                    batches = list(itertools.chain.from_iterable(mini_shards))

            return batches

        if self._supports_prefetch:
            raise NotImplementedError("To be implemented")
        else:
            batches = return_full_batches(
                self.frozen_batches, self.seed + epoch, shuffle
            )
            batches = list(
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
        )
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        return CountingIterator(itr, start=offset)
