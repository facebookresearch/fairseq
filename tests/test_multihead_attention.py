# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from fairseq.modules.multihead_attention import MultiheadAttention


def test_mask_padding_parity():

    def old_padding_code(key_padding_mask, attn_mask):
        if attn_mask is not None:
            attn_mask = torch.cat(
                [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
            )
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    torch.zeros(key_padding_mask.size(0), 1).type_as(
                        key_padding_mask
                    ),
                ],
                dim=1,
            )
        return key_padding_mask, attn_mask

    # values don't matter for this test.
    mha = MultiheadAttention(
        embedding=8,
        num_heads=2,
        dropout=0.0,
        add_bias_kv=True,
        add_zero_attn=True,
    )

    key_padding_mask = torch.rand((8, 64))
    attn_mask = torch.rand((64, 64))

    kp_mask_orig, a_mask_orig = old_padding_code(key_padding_mask, attn_mask)
    kp_mask_new, a_mask_new = mha._pad_masks(key_padding_mask, attn_mask)

    assert kp_mask_orig.size() == kp_mask_new.size()
    assert a_mask_orig.size() == a_mask_new.size()
    assert torch.equal(kp_mask_orig, kp_mask_new)
    assert torch.equal(a_mask_orig, a_mask_new)


def test_add_bias_parity():
    # values don't matter for this test.
    mha = MultiheadAttention(
        embedding=8,
        num_heads=2,
        dropout=0.0,
        add_bias_kv=True,
        add_zero_attn=True,
    )

    def old_bias_code(k, v, key_padding_mask, attn_mask, bsz):
        k = torch.cat([k, mha.bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, mha.bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = torch.cat(
                [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
            )
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                ],
                dim=1,
            )
        return k, v, key_padding_mask, attn_mask

    seq_len = 64
    bsz = 8
    embedding = 8
    key_padding_mask = torch.rand((bsz, seq_len))
    attn_mask = torch.rand((seq_len, seq_len))
    k = torch.rand((seq_len, bsz, embedding))
    v = torch.rand((seq_len, bsz, embedding))

    k_orig, v_orig, kp_mask_orig, a_mask_orig = old_bias_code(k, v, key_padding_mask, attn_mask, bsz)
    k_new, v_new, kp_mask_new, a_mask_new = mha._add_bias(k, v, key_padding_mask, attn_mask, bsz)

    assert torch.equal(k_orig, k_new)
    assert torch.equal(v_orig, v_new)
    assert torch.equal(kp_mask_orig, kp_mask_new)
    assert torch.equal(a_mask_orig, a_mask_new)


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
            # prev_key_padding_mask already full
            (
                torch.tensor([[0, 1, 0, 1]]).bool(),
                None,
                torch.tensor([[0, 1, 0, 1]]).bool(),
            ),
            # key_padding_mask already full
            (
                None,
                torch.tensor([[0, 1, 0, 1]]).bool(),
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

    def test_pruning_heads(self):
        embed_dim = 768
        num_heads = 12
        num_heads_to_keep = 8
        dummy_input = torch.randn(32, 2, embed_dim)
        mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        reserve_head_index = mha._get_reserve_head_index(
            num_heads_to_keep=num_heads_to_keep
        )
        mha._adaptive_prune_heads(reserve_head_index=reserve_head_index)
        mha._set_skip_embed_dim_check()
        mha(query=dummy_input, key=dummy_input, value=dummy_input)
        self.assertEqual(mha.head_dim, embed_dim / num_heads)
        self.assertEqual(mha.num_heads, num_heads_to_keep)


if __name__ == "__main__":
    unittest.main()
