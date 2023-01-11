import argparse
import unittest
from typing import Any, Dict

import torch
from examples.simultaneous_translation.models import (
    transformer_monotonic_attention
)


from tests.test_roberta import FakeTask


DEFAULT_CONFIG = {
    "attention_eps": 1e-6,
    "mass_preservation": True,
    "noise_type": "flat",
    "noise_mean": 0.0,
    "noise_var": 1.0,
    "energy_bias_init": -2,
    "energy_bias": True
}


PAD_INDEX = 1


def generate_config(overrides_kv):
    new_dict = {key: value for key, value in DEFAULT_CONFIG.items()}
    for key, value in overrides_kv.items():
        new_dict[key] = value
    return new_dict


def make_sample_with_padding(longer_src=False) -> Dict[str, Any]:
    tokens_1 = torch.LongTensor(
        [
            [2, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 2],
            [
                2, 11, 12, 14, 15, 10, 11, 12, 13, 14, 15, 2,
                PAD_INDEX, PAD_INDEX
            ],
        ]
    )
    tokens_2 = torch.LongTensor(
        [
            [2, 11, 12, 13, 14, 2, PAD_INDEX, PAD_INDEX],
            [2, 11, 22, 33, 2, PAD_INDEX, PAD_INDEX, PAD_INDEX]
        ]
    )
    if longer_src:
        src_tokens = tokens_1[:, 1:]
        prev_output_tokens = tokens_2
    else:
        src_tokens = tokens_2[:, 1:8]
        prev_output_tokens = tokens_1

    src_lengths = src_tokens.ne(PAD_INDEX).sum(dim=1).long()

    sample = {
        "net_input": {
            "src_tokens": src_tokens,
            "prev_output_tokens": prev_output_tokens,
            "src_lengths": src_lengths,
        },
        "target": prev_output_tokens[:, 1:],
    }
    return sample


def build_transformer_monotonic_attention(**extra_args: Any):
    overrides = {
        # Use characteristics dimensions
        "encoder_embed_dim": 12,
        "encoder_ffn_embed_dim": 14,
        "decoder_embed_dim": 12,
        "decoder_ffn_embed_dim": 14,
        # Disable dropout so we have comparable tests.
        "dropout": 0,
        "attention_dropout": 0,
        "activation_dropout": 0,
        "encoder_layerdrop": 0,
    }
    overrides.update(extra_args)
    # Overrides the defaults from the parser
    args = argparse.Namespace(**overrides)
    transformer_monotonic_attention.monotonic_tiny_architecture(args)

    torch.manual_seed(0)
    task = FakeTask(args)
    return (
        transformer_monotonic_attention
        .TransformerModelSimulTrans
        .build_model(args, task)
    )


def expected_alignment_formula(
    p_choose,
    mass_perservation=True,
    padding_mask=None
):
    # Online and Linear-Time Attention by Enforcing Monotonic Alignments
    # https://arxiv.org/pdf/1704.00784.pdf
    # Eq 18, 19
    bsz, tgt_len, src_len = p_choose.size()
    alpha = torch.zeros_like(p_choose)

    if padding_mask is not None:
        bsz_pad = padding_mask.size(0)
        num_heads = int(bsz / bsz_pad)
        padding_mask = (
            padding_mask
            .unsqueeze(1)
            .expand([bsz_pad, num_heads, src_len])
            .contiguous()
            .view(-1, src_len)
        )

    p_choose = p_choose.masked_fill(padding_mask.unsqueeze(1), 0)

    for bsz_i in range(bsz):
        for i in range(tgt_len):
            for j in range(src_len):
                if i == 0:
                    if j == 0:
                        # First source token
                        alpha[bsz_i, i, j] = p_choose[bsz_i, i, j]
                    else:
                        # First target token
                        alpha[bsz_i, i, j] = (
                            p_choose[bsz_i, i, j]
                            * torch.prod(
                                1 - p_choose[bsz_i, i, :j]
                            )
                        )
                else:
                    alpha[bsz_i, i, j] = alpha[bsz_i, i - 1, j]
                    for k in range(j):
                        alpha[bsz_i, i, j] += (
                            alpha[bsz_i, i - 1, k]
                            * torch.prod(
                                1 - p_choose[bsz_i, i, k:j]
                            )
                        )
                    alpha[bsz_i, i, j] *= p_choose[bsz_i, i, j]

    alpha = alpha.masked_fill(padding_mask.unsqueeze(1), 0)

    if mass_perservation:
        alpha = mass_perservation_formula(alpha, False, padding_mask)

    return alpha


def mass_perservation_formula(alpha, left_padding=False, padding_mask=None):
    if padding_mask is None or alpha.size(-1) == 1:
        if alpha.size(-1) > 1:
            alpha[:, :, -1] = 1 - alpha[:, :, :-1].sum(dim=-1)
        return alpha

    src_lens = (padding_mask.logical_not()).sum(dim=1).long()

    bsz, tgt_len, src_len = alpha.size()

    assert (
        not left_padding
        or (left_padding and (not padding_mask[:, 0].any()))
    )

    alpha = alpha.masked_fill(padding_mask.unsqueeze(1), 0)

    for bsz_i in range(bsz):
        if left_padding:
            alpha[bsz_i, :, -1] = (
                1 - alpha[bsz_i, :, :-1].sum(dim=-1)
            )
        else:
            alpha[bsz_i, :, src_lens[bsz_i] - 1] = (
                1 - alpha[bsz_i, :, :src_lens[bsz_i] - 1].sum(dim=-1)
            )

    return alpha


def expected_soft_attention_formula(
    alpha,
    soft_energy,
    padding_mask=None,
    chunksize=1e10,
):
    # Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    # https://arxiv.org/pdf/1906.05218.pdf
    # Eq 14

    # Monotonic Chunkwise Attention
    # https://arxiv.org/abs/1712.05382
    # Eq 17
    bsz, tgt_len, src_len = alpha.size()
    beta = torch.zeros_like(alpha)

    if padding_mask is not None:
        bsz_pad = padding_mask.size(0)
        num_heads = int(bsz / bsz_pad)
        # Expanding for potential head dimension
        padding_mask = (
            padding_mask
            .unsqueeze(1)
            .expand([bsz_pad, num_heads, src_len])
            .contiguous()
            .view(-1, src_len)
        )
        soft_energy = soft_energy.masked_fill(padding_mask.unsqueeze(1), float('-inf'))

    for bsz_i in range(bsz):
        for i in range(tgt_len):
            for j in range(src_len):
                for k in range(j, min([src_len, j + chunksize])):
                    if not padding_mask[bsz_i, j]:
                        beta[bsz_i, i, j] += (
                            alpha[bsz_i, i, k] * torch.exp(soft_energy[bsz_i, i, j])
                            / torch.sum(torch.exp(soft_energy[bsz_i, i, max([0, k - chunksize + 1]):k + 1]))
                        )
    return beta


class MonotonicAttentionTestAbstractClass(object):
    def test_forward(self):
        sample = make_sample_with_padding()
        out, _ = self.model.forward(**sample["net_input"])
        loss = out.sum()
        loss.backward()

    def test_p_choose(self):
        sample = make_sample_with_padding()
        _, extra_out = self.model.forward(**sample["net_input"])
        for item in extra_out.attn_list:
            p_choose = item["p_choose"]
            self.assertTrue(p_choose.le(1.0).all())
            self.assertTrue(p_choose.ge(0.0).all())

    def test_expected_alignment(self):
        for longer_src in [True, False]:
            sample = make_sample_with_padding(longer_src)
            _, extra_out = self.model.forward(**sample["net_input"])
            for item in extra_out.attn_list:
                p_choose = item["p_choose"]
                alpha_system = item["alpha"]
                self.assertTrue(p_choose.size() == alpha_system.size())
                bsz, num_head, tgt_len, src_len = alpha_system.size()
                alpha_system = alpha_system.view(-1, tgt_len, src_len)
                p_choose = p_choose.view(-1, tgt_len, src_len)

                alpha_real = expected_alignment_formula(
                    p_choose,
                    self.model.decoder.layers[0].encoder_attn.mass_preservation,
                    sample["net_input"]["src_tokens"].eq(PAD_INDEX)
                )

                self.assertTrue(
                    torch.abs(alpha_system - alpha_real).le(5e-5).all(),
                )


class HardMonotonicAttentionTestCase(
    unittest.TestCase,
    MonotonicAttentionTestAbstractClass
):
    def setUp(self):
        self.model = build_transformer_monotonic_attention(
            **generate_config({"simul_type": "hard_aligned"})
        )


class InfiniteLookbackTestCase(
    unittest.TestCase,
    MonotonicAttentionTestAbstractClass
):
    def setUp(self):
        self.model = build_transformer_monotonic_attention(
            **generate_config(
                {
                    "simul_type": "infinite_lookback"
                }
            )
        )
        self.model.train()

    def test_fp16_for_long_input(self):
        sample = {
            "net_input": {
                "src_tokens": torch.LongTensor([7] * 1000 + [2]).cuda().unsqueeze(0),
                "prev_output_tokens": torch.LongTensor([7] * 1000 + [2]).cuda().unsqueeze(0),
                "src_lengths": torch.LongTensor([1000]).cuda(),
            },
            "target": torch.LongTensor([2] + [7] * 1000).unsqueeze(0).cuda()
        }
        self.model.cuda().half()
        _, extra_out = self.model.forward(**sample["net_input"])
        for item in extra_out.attn_list:
            for key in ["p_choose", "alpha", "beta", "soft_energy"]:
                self.assertFalse(torch.isnan(item[key]).any())

    def test_expected_attention(self):
        for longer_src in [True, False]:
            sample = make_sample_with_padding(longer_src)
            _, extra_out = self.model.forward(**sample["net_input"])
            for item in extra_out.attn_list:
                p_choose = item["p_choose"]
                alpha_system = item["alpha"]
                beta_system = item["beta"]
                soft_energy_system = item["soft_energy"]
                self.assertTrue(beta_system.size() == alpha_system.size())
                self.assertTrue(p_choose.size() == alpha_system.size())

                bsz, num_head, tgt_len, src_len = alpha_system.size()

                alpha_system = alpha_system.view(-1, tgt_len, src_len)
                beta_system = beta_system.view(-1, tgt_len, src_len)
                p_choose = p_choose.view(-1, tgt_len, src_len)
                soft_energy_system = soft_energy_system.view(-1, tgt_len, src_len)

                alpha_real = expected_alignment_formula(
                    p_choose,
                    self.model.decoder.layers[0].encoder_attn.mass_preservation,
                    sample["net_input"]["src_tokens"].eq(PAD_INDEX)
                )

                beta_real = expected_soft_attention_formula(
                    alpha_real,
                    soft_energy_system,
                    sample["net_input"]["src_tokens"].eq(PAD_INDEX),
                    chunksize=getattr(
                        self.model.decoder.layers[0].encoder_attn,
                        "chunk_size",
                        int(1e10)
                    ) or int(1e10)
                )

                self.assertTrue(
                    torch.abs(beta_system - beta_real).le(1e-5).all(),
                )


class ChunkwiswTestCase(
    InfiniteLookbackTestCase
):
    def setUp(self):
        self.model = build_transformer_monotonic_attention(
            **generate_config(
                {
                    "simul_type": "chunkwise",
                    "mocha_chunk_size": 3
                }
            )
        )


class WaitkTestCase(InfiniteLookbackTestCase):
    def setUp(self):
        self.model = build_transformer_monotonic_attention(
            **generate_config(
                {
                    "simul_type": "waitk",
                    "waitk_lagging": 3,
                }
            )
        )

    def check_waitk(self, p_choose, lagging, padding_mask):
        bsz, tgt_len, src_len = p_choose.size()
        for bsz_i in range(bsz):
            for i in range(tgt_len):
                for j in range(src_len):
                    if not padding_mask[bsz_i, j]:
                        if j - i == lagging - 1:
                            self.assertTrue(p_choose[bsz_i, i, j] == 1)
                        else:
                            self.assertTrue(p_choose[bsz_i, i, j] == 0)

    def test_waitk_p_choose(self):
        for longer_src in [True, False]:
            for k in [1, 3, 10, 20, 100]:
                sample = make_sample_with_padding(longer_src)
                model = build_transformer_monotonic_attention(
                    **generate_config(
                        {
                            "simul_type": "waitk",
                            "waitk_lagging": k,
                        }
                    )
                )
                model.train()
                _, extra_out = model.forward(**sample["net_input"])
                for item in extra_out.attn_list:
                    p_choose = item["p_choose"]
                    bsz, num_heads, tgt_len, src_len = p_choose.size()
                    padding_mask = sample["net_input"]["src_tokens"].eq(PAD_INDEX)
                    padding_mask = (
                        padding_mask
                        .unsqueeze(1)
                        .expand([bsz, num_heads, src_len])
                        .contiguous()
                        .view(-1, src_len)
                    )
                    p_choose = p_choose.view(bsz * num_heads, tgt_len, src_len)
                    self.check_waitk(p_choose, k, padding_mask)
