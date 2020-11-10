# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from fairseq.modules.multihead_attention import MultiheadAttention


class TestMultiheadAttention(unittest.TestCase):
    def test_append_prev_key_padding_mask(self):
        bsz = 1
        src_len = 4

        cases = [
            # no padding mask
            (None, None, None),
            # current padding mask only
            (
                torch.tensor([[1]]).bool(),
                None,
                torch.tensor([[0, 0, 0, 1]]).bool(),
            ),
            # previous padding mask only
            (
                None,
                torch.tensor([[0, 1, 0]]).bool(),
                torch.tensor([[0, 1, 0, 0]]).bool(),
            ),
            # both padding masks
            (
                torch.tensor([[1]]).bool(),
                torch.tensor([[0, 1, 0]]).bool(),
                torch.tensor([[0, 1, 0, 1]]).bool(),
            ),
        ]
        for c in cases:
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                c[0],
                c[1],
                batch_size=bsz,
                src_len=src_len,
                static_kv=False,
            )

            if key_padding_mask is not None:
                self.assertTrue(
                    torch.all(torch.eq(key_padding_mask, c[2])),
                    f"Unexpected resultant key padding mask: {key_padding_mask}"
                    f" given current: {c[0]} and previous: {c[1]}",
                )
                self.assertEqual(key_padding_mask.size(0), bsz)
                self.assertEqual(key_padding_mask.size(1), src_len)
            else:
                self.assertIsNone(c[2])


if __name__ == "__main__":
    unittest.main()
