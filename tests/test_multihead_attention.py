# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

import pytest
import torch

from fairseq.modules.multihead_attention import MultiheadAttention, _mask_for_xformers

BATCH = [20, 41, 97]
SEQ = [64]
EMB = [48]
HEADS = [4]
DROP = 0.1
DEVICE = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
ATTN_MASK_DTYPE = [None, torch.uint8, torch.bool, torch.float]
KEY_PADDING_MASK_DTYPE = [None, torch.uint8, torch.bool]


# FIXME: some tests fail when decimal=2, fix this and set decimal to 2
def assert_almost_equal(x, y, decimal=1, err_msg=""):
    import numpy.testing as npt

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    npt.assert_array_almost_equal(x, y, err_msg=err_msg, decimal=decimal)


def _reset_seeds():
    torch.manual_seed(0)
    torch.random.manual_seed(0)
    random.seed(0)
    torch.cuda.manual_seed_all(0)


def _get_mask(to_dtype: torch.dtype, dim0: int, dim1: int):
    if to_dtype == torch.float:
        mask = torch.randint(0, 2, (dim0, dim1)).to(dtype=torch.bool)
        return mask.to(dtype=to_dtype).masked_fill(mask, -float("inf"))
    return torch.randint(0, 2, (dim0, dim1)).to(dtype=to_dtype)


def test_mask_for_xformers():
    # Additive Mask
    m_float_add = torch.tensor([float("-inf"), 0]).to(torch.float)
    m_float_add_flipped = torch.tensor([0, float("-inf")]).to(torch.float)
    m_float16_add = torch.tensor([float("-inf"), 0]).to(torch.float16)
    m_float16_add_flipped = torch.tensor([0, float("-inf")]).to(torch.float16)
    m_uint = torch.tensor([1, 0]).to(torch.uint8)
    m_uint_flipped = torch.tensor([0, 1]).to(torch.uint8)
    m_bool = torch.tensor([False, True])

    assert torch.equal(_mask_for_xformers(m_float_add), m_float_add)
    assert torch.equal(_mask_for_xformers(m_float16_add), m_float16_add)
    assert torch.equal(_mask_for_xformers(m_uint), m_uint_flipped)
    assert torch.equal(_mask_for_xformers(m_bool), ~m_bool)

    assert torch.equal(
        _mask_for_xformers(m_float_add, to_dtype=torch.float16), m_float16_add
    )
    assert torch.equal(
        _mask_for_xformers(m_float_add, to_dtype=torch.float), m_float_add
    )
    assert torch.equal(_mask_for_xformers(m_float_add, to_dtype=torch.bool), m_bool)
    assert torch.equal(
        _mask_for_xformers(m_float_add, to_dtype=torch.uint8), m_uint_flipped
    )

    assert torch.equal(
        _mask_for_xformers(m_float16_add, to_dtype=torch.float16), m_float16_add
    )
    assert torch.equal(
        _mask_for_xformers(m_float16_add, to_dtype=torch.float), m_float_add
    )
    assert torch.equal(_mask_for_xformers(m_float16_add, to_dtype=torch.bool), m_bool)
    assert torch.equal(
        _mask_for_xformers(m_float16_add, to_dtype=torch.uint8), m_uint_flipped
    )

    assert torch.equal(
        _mask_for_xformers(m_bool, to_dtype=torch.float16), m_float16_add_flipped
    )
    assert torch.equal(
        _mask_for_xformers(m_bool, to_dtype=torch.float), m_float_add_flipped
    )
    assert torch.equal(_mask_for_xformers(m_bool, to_dtype=torch.bool), ~m_bool)
    assert torch.equal(_mask_for_xformers(m_bool, to_dtype=torch.uint8), m_uint)

    assert torch.equal(
        _mask_for_xformers(m_uint, to_dtype=torch.float16), m_float16_add
    )
    assert torch.equal(_mask_for_xformers(m_uint, to_dtype=torch.float), m_float_add)
    assert torch.equal(_mask_for_xformers(m_uint, to_dtype=torch.bool), m_bool)
    assert torch.equal(_mask_for_xformers(m_uint, to_dtype=torch.uint8), m_uint_flipped)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="blocksparse requires gpu")
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("add_zero_attn", [False])
@pytest.mark.parametrize("batch_size", [20])
@pytest.mark.parametrize("embedding", [64])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("num_heads", [4])
def test_xformers_blocksparse_parity(
    device,
    add_zero_attn,
    batch_size,
    embedding,
    seq_len,
    num_heads,
):

    xformers_att_config = '{"name": "scaled_dot_product"}'
    xformers_blocksparse_blocksize = 16
    xformers_blocksparse_layout = torch.ones(
        seq_len // xformers_blocksparse_blocksize,
        seq_len // xformers_blocksparse_blocksize,
        dtype=torch.int32,
    )

    q = torch.rand(seq_len, batch_size, embedding).to(device).half()
    q.requires_grad = True
    k = torch.rand(seq_len, batch_size, embedding).to(device).half()
    k.requires_grad = True
    v = torch.rand(seq_len, batch_size, embedding).to(device).half()
    v.requires_grad = True

    q_ = q.detach().clone().half()
    q_.requires_grad = True
    k_ = k.detach().clone().half()
    k_.requires_grad = True
    v_ = v.detach().clone().half()
    v_.requires_grad = True

    _reset_seeds()
    xf_blocksparse_mha = (
        MultiheadAttention(
            embedding,
            num_heads,
            dropout=0.0,
            add_zero_attn=add_zero_attn,
            xformers_att_config=xformers_att_config,
            xformers_blocksparse_layout=xformers_blocksparse_layout,
            xformers_blocksparse_blocksize=xformers_blocksparse_blocksize,
        )
        .to(device)
        .half()
    )

    xf_blocksparse_output, _ = xf_blocksparse_mha(
        q,
        k,
        v,
    )

    _reset_seeds()
    xformers_mha = (
        MultiheadAttention(
            embedding,
            num_heads,
            dropout=0.0,
            add_zero_attn=add_zero_attn,
            xformers_att_config=xformers_att_config,
            xformers_blocksparse_layout=None,
        )
        .to(device)
        .half()
    )

    xformers_output, _ = xformers_mha(
        q_,
        k_,
        v_,
    )

    # # account for when nan != nan
    rand = random.uniform(0, 1)
    xformers_output = xformers_output.masked_fill(xformers_output.isnan(), rand)
    xf_blocksparse_output = xf_blocksparse_output.masked_fill(
        xf_blocksparse_output.isnan(), rand
    )

    assert_almost_equal(xformers_output, xf_blocksparse_output)

    loss_blocksparse = torch.norm(xformers_output)
    loss_original = torch.norm(xf_blocksparse_output)
    loss_blocksparse.backward()
    loss_original.backward()

    q.masked_fill(q.isnan(), rand)
    q_.masked_fill(q_.isnan(), rand)
    k.masked_fill(k.isnan(), rand)
    k_.masked_fill(k_.isnan(), rand)
    v.masked_fill(v.isnan(), rand)
    v_.masked_fill(v_.isnan(), rand)

    assert_almost_equal(q.grad, q_.grad)
    assert_almost_equal(k.grad, k_.grad)
    assert_almost_equal(v.grad, v_.grad)


@pytest.mark.parametrize("device", DEVICE)
@pytest.mark.parametrize("attn_dtype", ATTN_MASK_DTYPE)
@pytest.mark.parametrize("key_padding_dtype", KEY_PADDING_MASK_DTYPE)
@pytest.mark.parametrize("add_bias_kv", [True, False])
@pytest.mark.parametrize("add_zero_attn", [True, False])
# TODO: test with static_kv True
@pytest.mark.parametrize("static_kv", [False])
@pytest.mark.parametrize("batch_size", BATCH)
@pytest.mark.parametrize("embedding", EMB)
@pytest.mark.parametrize("seq_len", SEQ)
@pytest.mark.parametrize("num_heads", HEADS)
def test_xformers_single_forward_parity(
    device,
    attn_dtype,
    key_padding_dtype,
    add_bias_kv,
    add_zero_attn,
    static_kv,
    batch_size,
    embedding,
    seq_len,
    num_heads,
):

    xformers_att_config = '{"name": "scaled_dot_product"}'

    attn_mask = (
        None
        if attn_dtype is None
        else _get_mask(to_dtype=attn_dtype, dim0=seq_len, dim1=seq_len).to(device)
    )
    key_padding_mask = (
        None
        if key_padding_dtype is None
        else _get_mask(to_dtype=key_padding_dtype, dim0=batch_size, dim1=seq_len).to(
            device
        )
    )

    q = torch.rand(seq_len, batch_size, embedding).to(device)
    q.requires_grad = True
    k = torch.rand(seq_len, batch_size, embedding).to(device)
    k.requires_grad = True
    v = torch.rand(seq_len, batch_size, embedding).to(device)
    v.requires_grad = True

    q_ = q.detach().clone()
    q_.requires_grad = True
    k_ = k.detach().clone()
    k_.requires_grad = True
    v_ = v.detach().clone()
    v_.requires_grad = True

    # TODO: dropouts in the two implementations lead to different entries dropped.
    _reset_seeds()
    xformers_mha = MultiheadAttention(
        embedding,
        num_heads,
        dropout=0.0,
        xformers_att_config=xformers_att_config,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
    ).to(device)
    xformers_output, _ = xformers_mha(
        q,
        k,
        v,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        static_kv=static_kv,
    )

    _reset_seeds()
    original_mha = MultiheadAttention(
        embedding,
        num_heads,
        dropout=0.0,
        xformers_att_config=None,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
    ).to(device)
    original_output, _ = original_mha(
        q_,
        k_,
        v_,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        static_kv=static_kv,
    )

    # account for when nan != nan
    if xformers_output.isnan().any() or original_output.isnan().any():
        rand = random.uniform(0, 1)
        xformers_output = xformers_output.masked_fill(xformers_output.isnan(), rand)
        original_output = original_output.masked_fill(original_output.isnan(), rand)

    # torch.equal works for cpu, on cuda allclose is needed.
    assert torch.allclose(
        xformers_output, original_output, atol=1e-06
    ), f"max diff is {torch.max(torch.abs(xformers_output - original_output))}"

    loss_xformers = torch.norm(xformers_output)
    loss_original = torch.norm(original_output)
    loss_xformers.backward()
    loss_original.backward()

    # torch.equal works for cpu, on cuda allclose is needed.
    assert torch.allclose(
        q.grad, q_.grad
    ), f"max diff is {torch.max(torch.abs(q.grad - q_.grad))}"
    assert torch.allclose(
        k.grad, k_.grad
    ), f"max diff is {torch.max(torch.abs(k.grad - k_.grad))}"
    assert torch.allclose(
        v.grad, v_.grad
    ), f"max diff is {torch.max(torch.abs(v.grad - v_.grad))}"


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
                    torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                ],
                dim=1,
            )
        return key_padding_mask, attn_mask

    # values don't matter for this test.
    mha = MultiheadAttention(
        embed_dim=8,
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
        embed_dim=8,
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

    k_orig, v_orig, kp_mask_orig, a_mask_orig = old_bias_code(
        k, v, key_padding_mask, attn_mask, bsz
    )
    k_new, v_new, kp_mask_new, a_mask_new = mha._add_bias(
        k, v, key_padding_mask, attn_mask, bsz
    )

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
