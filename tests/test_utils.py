# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import unittest

import torch
from torch.autograd import Variable

from fairseq import utils


class TestUtils(unittest.TestCase):

    def test_convert_padding_direction(self):
        pad = 1
        left_pad = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [1, 7, 8, 9, 10],
            [1, 1, 1, 11, 12],
        ])
        right_pad = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [7, 8, 9, 10, 1],
            [11, 12, 1, 1, 1],
        ])
        lengths = torch.LongTensor([5, 4, 2])

        self.assertAlmostEqual(
            right_pad,
            utils.convert_padding_direction(
                left_pad,
                lengths,
                pad,
                left_to_right=True,
            ),
        )
        self.assertAlmostEqual(
            left_pad,
            utils.convert_padding_direction(
                right_pad,
                lengths,
                pad,
                right_to_left=True,
            ),
        )

    def test_make_variable(self):
        t = [{'k': torch.rand(5, 5)}]

        v = utils.make_variable(t)[0]['k']
        self.assertTrue(isinstance(v, Variable))
        self.assertFalse(v.data.is_cuda)

        v = utils.make_variable(t, cuda=True)[0]['k']
        self.assertEqual(v.data.is_cuda, torch.cuda.is_available())

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess(utils.item((t1 - t2).abs().max()), 1e-4)


if __name__ == '__main__':
    unittest.main()
