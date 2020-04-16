# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

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

        self.assertAlmostEqual(
            right_pad,
            utils.convert_padding_direction(
                left_pad,
                pad,
                left_to_right=True,
            ),
        )
        self.assertAlmostEqual(
            left_pad,
            utils.convert_padding_direction(
                right_pad,
                pad,
                right_to_left=True,
            ),
        )

    def test_make_positions(self):
        pad = 1
        left_pad_input = torch.LongTensor([
            [9, 9, 9, 9, 9],
            [1, 9, 9, 9, 9],
            [1, 1, 1, 9, 9],
        ])
        left_pad_output = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 2, 3],
        ])
        right_pad_input = torch.LongTensor([
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 1],
            [9, 9, 1, 1, 1],
        ])
        right_pad_output = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [2, 3, 4, 5, 1],
            [2, 3, 1, 1, 1],
        ])

        self.assertAlmostEqual(
            left_pad_output,
            utils.make_positions(left_pad_input, pad),
        )
        self.assertAlmostEqual(
            right_pad_output,
            utils.make_positions(right_pad_input, pad),
        )

    def test_clip_grad_norm_(self):
        params = torch.nn.Parameter(torch.zeros(5)).requires_grad_(False)
        grad_norm = utils.clip_grad_norm_(params, 1.0)
        self.assertTrue(torch.is_tensor(grad_norm))
        self.assertEqual(grad_norm, 0.0)

        params = [torch.nn.Parameter(torch.zeros(5)) for i in range(3)]
        for p in params:
            p.grad = torch.full((5,), fill_value=2)
        grad_norm = utils.clip_grad_norm_(params, 1.0)
        exp_grad_norm = torch.full((15,), fill_value=2).norm()
        self.assertTrue(torch.is_tensor(grad_norm))
        self.assertEqual(grad_norm, exp_grad_norm)

        grad_norm = utils.clip_grad_norm_(params, 1.0)
        self.assertAlmostEqual(grad_norm, torch.tensor(1.0))

    def test_resolve_max_positions_with_tuple(self):
        resolved = utils.resolve_max_positions(None, (2000, 100, 2000), 12000)
        self.assertEqual(resolved, (2000, 100, 2000))

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess(utils.item((t1 - t2).abs().max()), 1e-4)


if __name__ == '__main__':
    unittest.main()
