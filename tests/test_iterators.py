# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from fairseq.data import iterators


class TestIterators(unittest.TestCase):
    def test_counting_iterator_index(self, ref=None, itr=None):
        # Test the indexing functionality of CountingIterator
        if ref is None:
            assert itr is None
            ref = list(range(10))
            itr = iterators.CountingIterator(ref)
        else:
            assert len(ref) == 10
            assert itr is not None

        self.assertTrue(itr.has_next())
        self.assertEqual(itr.n, 0)
        self.assertEqual(next(itr), ref[0])
        self.assertEqual(itr.n, 1)
        self.assertEqual(next(itr), ref[1])
        self.assertEqual(itr.n, 2)
        itr.skip(3)
        self.assertEqual(itr.n, 5)
        self.assertEqual(next(itr), ref[5])
        itr.skip(2)
        self.assertEqual(itr.n, 8)
        self.assertEqual(list(itr), [ref[8], ref[9]])
        self.assertFalse(itr.has_next())

    def test_counting_iterator_length_mismatch(self):
        ref = list(range(10))
        # When the underlying iterable is longer than the CountingIterator,
        # the remaining items in the iterable should be ignored
        itr = iterators.CountingIterator(ref, total=8)
        self.assertEqual(list(itr), ref[:8])
        # When the underlying iterable is shorter than the CountingIterator,
        # raise an IndexError when the underlying iterable is exhausted
        itr = iterators.CountingIterator(ref, total=12)
        self.assertRaises(IndexError, list, itr)

    def test_counting_iterator_take(self):
        # Test the "take" method of CountingIterator
        ref = list(range(10))
        itr = iterators.CountingIterator(ref)
        itr.take(5)
        self.assertEqual(len(itr), len(list(iter(itr))))
        self.assertEqual(len(itr), 5)

        itr = iterators.CountingIterator(ref)
        itr.take(5)
        self.assertEqual(next(itr), ref[0])
        self.assertEqual(next(itr), ref[1])
        itr.skip(2)
        self.assertEqual(next(itr), ref[4])
        self.assertFalse(itr.has_next())

    def test_grouped_iterator(self):
        # test correctness
        x = list(range(10))
        itr = iterators.GroupedIterator(x, 1)
        self.assertEqual(list(itr), [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
        itr = iterators.GroupedIterator(x, 4)
        self.assertEqual(list(itr), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        itr = iterators.GroupedIterator(x, 5)
        self.assertEqual(list(itr), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

        # test the GroupIterator also works correctly as a CountingIterator
        x = list(range(30))
        ref = list(iterators.GroupedIterator(x, 3))
        itr = iterators.GroupedIterator(x, 3)
        self.test_counting_iterator_index(ref, itr)

    def test_sharded_iterator(self):
        # test correctness
        x = list(range(10))
        itr = iterators.ShardedIterator(x, num_shards=1, shard_id=0)
        self.assertEqual(list(itr), x)
        itr = iterators.ShardedIterator(x, num_shards=2, shard_id=0)
        self.assertEqual(list(itr), [0, 2, 4, 6, 8])
        itr = iterators.ShardedIterator(x, num_shards=2, shard_id=1)
        self.assertEqual(list(itr), [1, 3, 5, 7, 9])
        itr = iterators.ShardedIterator(x, num_shards=3, shard_id=0)
        self.assertEqual(list(itr), [0, 3, 6, 9])
        itr = iterators.ShardedIterator(x, num_shards=3, shard_id=1)
        self.assertEqual(list(itr), [1, 4, 7, None])
        itr = iterators.ShardedIterator(x, num_shards=3, shard_id=2)
        self.assertEqual(list(itr), [2, 5, 8, None])

        # test CountingIterator functionality
        x = list(range(30))
        ref = list(iterators.ShardedIterator(x, num_shards=3, shard_id=0))
        itr = iterators.ShardedIterator(x, num_shards=3, shard_id=0)
        self.test_counting_iterator_index(ref, itr)

    def test_counting_iterator_buffered_iterator_take(self):
        ref = list(range(10))
        buffered_itr = iterators.BufferedIterator(2, ref)
        itr = iterators.CountingIterator(buffered_itr)
        itr.take(5)
        self.assertEqual(len(itr), len(list(iter(itr))))
        self.assertEqual(len(itr), 5)

        buffered_itr = iterators.BufferedIterator(2, ref)
        itr = iterators.CountingIterator(buffered_itr)
        itr.take(5)
        self.assertEqual(len(buffered_itr), 5)
        self.assertEqual(len(list(iter(buffered_itr))), 5)

        buffered_itr = iterators.BufferedIterator(2, ref)
        itr = iterators.CountingIterator(buffered_itr)
        itr.take(5)
        self.assertEqual(next(itr), ref[0])
        self.assertEqual(next(itr), ref[1])
        itr.skip(2)
        self.assertEqual(next(itr), ref[4])
        self.assertFalse(itr.has_next())
        self.assertRaises(StopIteration, next, buffered_itr)

        ref = list(range(4, 10))
        buffered_itr = iterators.BufferedIterator(2, ref)
        itr = iterators.CountingIterator(buffered_itr, start=4)
        itr.take(5)
        self.assertEqual(len(itr), 5)
        self.assertEqual(len(buffered_itr), 1)
        self.assertEqual(next(itr), ref[0])
        self.assertFalse(itr.has_next())
        self.assertRaises(StopIteration, next, buffered_itr)


if __name__ == "__main__":
    unittest.main()
