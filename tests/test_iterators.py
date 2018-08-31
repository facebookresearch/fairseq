# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import unittest

from fairseq.data import iterators


class TestIterators(unittest.TestCase):

    def test_counting_iterator(self):
        x = list(range(10))
        itr = iterators.CountingIterator(x)
        self.assertTrue(itr.has_next())
        self.assertEqual(next(itr), 0)
        self.assertEqual(next(itr), 1)
        itr.skip(3)
        self.assertEqual(next(itr), 5)
        itr.skip(3)
        self.assertEqual(next(itr), 9)
        self.assertFalse(itr.has_next())


if __name__ == '__main__':
    unittest.main()
