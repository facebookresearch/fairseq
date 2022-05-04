# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random

import torch
from torch.utils import benchmark

from fairseq.modules.multihead_attention import MultiheadAttention

BATCH = [20, 41, 97]
SEQ = 64
EMB = 48
HEADS = 4
DROP = 0.1
DEVICE = torch.device("cuda")
ATTN_MASK_DTYPE = [torch.uint8, torch.bool, torch.float]
KEY_PADDING_MASK_DTYPE = [torch.uint8, torch.bool]


def _reset_seeds():
    torch.manual_seed(0)
    random.seed(0)


def _get_mask(to_dtype: torch.dtype, dim0: int, dim1: int):
    if to_dtype == torch.float:
        mask = torch.randint(0, 2, (dim0, dim1)).to(dtype=torch.bool)
        return mask.to(dtype=to_dtype).masked_fill(mask, -float("inf"))
    return torch.randint(0, 2, (dim0, dim1)).to(dtype=to_dtype)


def benchmark_multihead_attention(
    label="",
    attn_dtype=torch.uint8,
    key_padding_dtype=torch.uint8,
    add_bias_kv=False,
    add_zero_attn=False,
    static_kv=False,
    batch_size=20,
    embedding=EMB,
    seq_len=SEQ,
    num_heads=HEADS,
):

    results = []
    # device = torch.device("cuda")

    xformers_att_config = '{"name": "scaled_dot_product"}'

    attn_mask = _get_mask(to_dtype=attn_dtype, dim0=seq_len, dim1=seq_len)
    key_padding_mask = _get_mask(
        to_dtype=key_padding_dtype, dim0=batch_size, dim1=seq_len
    )

    q = torch.rand(seq_len, batch_size, embedding, requires_grad=True)
    k = torch.rand(seq_len, batch_size, embedding, requires_grad=True)
    v = torch.rand(seq_len, batch_size, embedding, requires_grad=True)

    _reset_seeds()

    original_mha = MultiheadAttention(
        embedding,
        num_heads,
        dropout=0.0,
        xformers_att_config=None,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
    )

    xformers_mha = MultiheadAttention(
        embedding,
        num_heads,
        dropout=0.0,
        xformers_att_config=xformers_att_config,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
    )

    def original_bench_fw(q, k, v, key_padding_mask, attn_mask, static_kv):
        original_mha(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            static_kv=static_kv,
        )

    def xformers_bench_fw(q, k, v, key_padding_mask, attn_mask, static_kv):
        xformers_mha(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            static_kv=static_kv,
        )

    def original_bench_fw_bw(q, k, v, key_padding_mask, attn_mask, static_kv):
        output, _ = original_mha(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            static_kv=static_kv,
        )
        loss = torch.norm(output)
        loss.backward()

    def xformers_bench_fw_bw(q, k, v, key_padding_mask, attn_mask, static_kv):
        output, _ = xformers_mha(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            static_kv=static_kv,
        )
        loss = torch.norm(output)
        loss.backward()

    fns = [
        original_bench_fw,
        xformers_bench_fw,
        original_bench_fw_bw,
        xformers_bench_fw_bw,
    ]

    for fn in fns:
        results.append(
            benchmark.Timer(
                stmt="fn(q, k, v, key_padding_mask, attn_mask, static_kv)",
                globals={
                    "q": q,
                    "k": k,
                    "v": v,
                    "key_padding_mask": key_padding_mask,
                    "attn_mask": attn_mask,
                    "static_kv": static_kv,
                    "fn": fn,
                },
                label="multihead fw + bw",
                sub_label=f"{fn.__name__}",
                description=label,
            ).blocked_autorange(min_run_time=1)
        )

    compare = benchmark.Compare(results)
    compare.print()


def run_benchmarks():
    for attn_dtype, key_padding_dtype, add_bias_kv, add_zero_attn in itertools.product(
        ATTN_MASK_DTYPE, KEY_PADDING_MASK_DTYPE, [True, False], [True, False]
    ):
        label = f"attn_dtype {attn_dtype}, key_padding_dtype {key_padding_dtype}, \
            add_bias_kv {add_bias_kv}, add_zero_attn {add_zero_attn}"
        benchmark_multihead_attention(
            label=label,
            attn_dtype=attn_dtype,
            key_padding_dtype=key_padding_dtype,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )


run_benchmarks()
