# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from fairseq.data import iterators


class TestIterators(unittest.TestCase):

    def test_counting_iterator(self, ref=None, itr=None):
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
        itr.skip(3)
        self.assertEqual(itr.n, 9)
        self.assertEqual(next(itr), ref[9])
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

        # test CountingIterator functionality
        x = list(range(30))
        ref = list(iterators.GroupedIterator(x, 3))
        itr = iterators.GroupedIterator(x, 3)
        self.test_counting_iterator(ref, itr)

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
        self.test_counting_iterator(ref, itr)


if __name__ == '__main__':
    unittest.main()
