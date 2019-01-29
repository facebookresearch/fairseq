# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import unittest

import torch

from fairseq.data import TokenBlockDataset

import tests.utils as test_utils


class TestTokenBlockDataset(unittest.TestCase):

    def _build_dataset(self, data, **kwargs):
        sizes = [len(x) for x in data]
        underlying_ds = test_utils.TestDataset(data)
        return TokenBlockDataset(underlying_ds, sizes, **kwargs)

    def test_eos_break_mode(self):
        data = [
            torch.LongTensor([5, 4, 3, 2, 1]),
            torch.LongTensor([1]),  # this should be filtered
            torch.LongTensor([8, 7, 6, 1]),
        ]
        ds = self._build_dataset(data, block_size=None, pad=0, eos=1, break_mode='eos')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [1])
        self.assertEqual(ds[2].tolist(), [8, 7, 6, 1])

        data = [
            torch.LongTensor([5, 4, 3, 2, 1]),
            torch.LongTensor([8, 7, 6, 1]),
            torch.LongTensor([1]),  # this should be filtered
        ]
        ds = self._build_dataset(data, block_size=None, pad=0, eos=1, break_mode='eos')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [8, 7, 6, 1])
        self.assertEqual(ds[2].tolist(), [1])

    def test_block_break_mode(self):
        data = [
            torch.LongTensor([5, 4, 3, 2, 1]),
            torch.LongTensor([8, 7, 6, 1]),
            torch.LongTensor([9, 1]),
        ]
        ds = self._build_dataset(data, block_size=3, pad=0, eos=1, break_mode='none')
        self.assertEqual(ds[0].tolist(), [5, 4, 3])
        self.assertEqual(ds[1].tolist(), [2, 1, 8])
        self.assertEqual(ds[2].tolist(), [7, 6, 1])
        self.assertEqual(ds[3].tolist(), [9, 1])

    def test_complete_break_mode(self):
        data = [
            torch.LongTensor([5, 4, 3, 2, 1]),
            torch.LongTensor([8, 7, 6, 1]),
            torch.LongTensor([9, 1]),
        ]
        ds = self._build_dataset(data, block_size=6, pad=0, eos=1, break_mode='complete')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [8, 7, 6, 1, 9, 1])

        data = [
            torch.LongTensor([4, 3, 2, 1]),
            torch.LongTensor([5, 1]),
            torch.LongTensor([1]),
            torch.LongTensor([6, 1]),
        ]
        ds = self._build_dataset(data, block_size=3, pad=0, eos=1, break_mode='complete')
        self.assertEqual(ds[0].tolist(), [4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [5, 1, 1])
        self.assertEqual(ds[2].tolist(), [6, 1])


if __name__ == "__main__":
    unittest.main()
