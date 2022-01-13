# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor
import torch.nn as nn

from examples.simultaneous_translation.utils.p_choose_strategy import (
    learnable_p_choose,
    waitk_p_choose
)

from examples.simultaneous_translation.utils.monotonic_attention import (
    expected_alignment_from_p_choose,
    expected_soft_attention,
    mass_preservation,
)
from fairseq.modules import MultiheadAttention

from . import register_monotonic_attention
from typing import Dict, Optional


@register_monotonic_attention("hard_aligned")
class MonotonicAttention(MultiheadAttention):
    """
    Abstract class of monotonic attentions
    """
    k_in_proj: Dict[str, nn.Linear]
    q_in_proj: Dict[str, nn.Linear]

    def __init__(self, args):
        super().__init__(
            embed_dim=args.decoder_embed_dim,
            num_heads=args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.soft_attention = False

        self.eps = getattr(args, "attention_eps", True)
        self.mass_preservation = getattr(args, "mass_preservation", True)

        self.noise_type = args.noise_type
        self.noise_mean = args.noise_mean
        self.noise_var = args.noise_var

        self.energy_bias_init = args.energy_bias_init
        self.energy_bias = (
            nn.Parameter(self.energy_bias_init * torch.ones([1]))
            if args.energy_bias is True
            else 0
        )

        self.k_in_proj = {"monotonic": self.k_proj}
        self.q_in_proj = {"monotonic": self.q_proj}
        self.chunk_size = None

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--no-mass-preservation', action="store_false",
                            dest="mass_preservation",
                            help='Do not stay on the last token when decoding')
        parser.add_argument('--mass-preservation', action="store_true",
                            dest="mass_preservation",
                            help='Stay on the last token when decoding')
        parser.set_defaults(mass_preservation=True)
        parser.add_argument('--noise-var', type=float, default=1.0,
                            help='Variance of discretness noise')
        parser.add_argument('--noise-mean', type=float, default=0.0,
                            help='Mean of discretness noise')
        parser.add_argument('--noise-type', type=str, default="flat",
                            help='Type of discretness noise')
        parser.add_argument('--energy-bias', action="store_true",
                            default=False,
                            help='Bias for energy')
        parser.add_argument('--energy-bias-init', type=float, default=-2.0,
                            help='Initial value of the bias for energy')
        parser.add_argument('--attention-eps', type=float, default=1e-6,
                            help='Epsilon when calculating expected attention')

    def energy_from_qk(
        self,
        query: Tensor,
        key: Tensor,
        energy_type: str,
        key_padding_mask: Optional[Tensor] = None,
        bias: int = 0
    ):
        """
        Compute energy from query and key
        q_func_value is a tuple looks like
        (q_proj_func, q_tensor)
        q_tensor size: bsz, tgt_len, emb_dim
        k_tensor size: bsz, src_len, emb_dim
        key_padding_mask size: bsz, src_len
        attn_mask: bsz, src_len
        """

        length, bsz, _ = query.size()
        q = self.q_in_proj[energy_type].forward(query)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        q = q * self.scaling
        length, bsz, _ = key.size()
        k = self.k_in_proj[energy_type].forward(key)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        energy = torch.bmm(q, k.transpose(1, 2)) + bias

        if key_padding_mask is not None:
            energy = energy.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool),
                - float("inf")
            )

        return energy

    def p_choose_from_qk(self, query, key, key_padding_mask, incremental_states=None):
        monotonic_energy = self.energy_from_qk(
            query,
            key,
            "monotonic",
            key_padding_mask=key_padding_mask,
            bias=self.energy_bias,
        )

        p_choose = learnable_p_choose(
            monotonic_energy,
            self.noise_mean,
            self.noise_var,
            self.training
        )
        return p_choose

    def p_choose(self, query, key, key_padding_mask, incremental_states=None):
        return self.p_choose_from_qk(self, query, key, key_padding_mask)

    def monotonic_attention_process_infer(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    ):
        """
        Monotonic attention at inference time
        Notice that this function is designed for simuleval not sequence_generator
        """
        assert query is not None
        assert key is not None

        if query.size(1) != 1:
            raise RuntimeError(
                "Simultaneous translation models don't support batch decoding."
            )
        # 1. compute stepwise probability
        p_choose = self.p_choose(
            query, key, None, incremental_state
        ).squeeze(1)

        # 2. Compute the alpha
        src_len = key.size(0)
        # Maximum steps allows in this iteration
        max_steps = src_len - 1 if self.mass_preservation else src_len
        monotonic_cache = self._get_monotonic_buffer(incremental_state)
        # Step for each head
        monotonic_step = monotonic_cache.get(
            'head_step',
            p_choose.new_zeros(1, self.num_heads).long()
        )
        assert monotonic_step is not None
        finish_read = monotonic_step.eq(max_steps)
        p_choose_i = torch.tensor(1)

        while finish_read.sum().item() < self.num_heads:
            # p_choose: self.num_heads, src_len
            # only choose the p at monotonic steps
            # p_choose_i: 1, self.num_heads
            p_choose_i = (
                p_choose.gather(
                    1,
                    monotonic_step
                    .clamp(0, src_len - 1),
                )
            )

            read_one_step = (
                (p_choose_i < 0.5)
                .type_as(monotonic_step)
                .masked_fill(finish_read, 0)
            )
            # 1 x bsz
            # sample actions on unfinished seq
            # 0 means stay, finish reading
            # 1 means leave, continue reading

            monotonic_step += read_one_step

            finish_read = monotonic_step.eq(max_steps) | (read_one_step == 0)

        # p_choose at last steps
        p_choose_i = (
            p_choose.gather(
                1,
                monotonic_step
                .clamp(0, src_len - 1),
            )
        )

        monotonic_cache["head_step"] = monotonic_step
        # Whether a head is looking for new input
        monotonic_cache["head_read"] = (
            monotonic_step.eq(max_steps) & (p_choose_i < 0.5)
        )
        self._set_monotonic_buffer(incremental_state, monotonic_cache)

        # 2. Update alpha
        alpha = (
            p_choose
            .new_zeros([self.num_heads, src_len])
            .scatter(
                1,
                (monotonic_step)
                .view(self.num_heads, 1).clamp(0, src_len - 1),
                1
            )
        )

        if not self.mass_preservation:
            alpha = alpha.masked_fill(
                (monotonic_step == max_steps)
                .view(self.num_heads, 1),
                0
            )

        # 4. Compute Beta
        if self.soft_attention:
            monotonic_step = monotonic_step.t()
            beta_mask = torch.arange(src_len).expand_as(alpha).gt(monotonic_step).unsqueeze(1)
            # If it's soft attention just do softmax on current context
            soft_energy = self.energy_from_qk(
                query,
                key,
                "soft"
            )
            beta = torch.nn.functional.softmax(
                soft_energy.masked_fill(beta_mask, -float("inf")), dim=-1
            )
            # It could happen that a head doesn't move at all
            beta = beta.masked_fill(monotonic_step.eq(0).unsqueeze(1), 0)
        else:
            # If it's hard attention just select the last state
            beta = alpha

        return p_choose, alpha, beta

    def monotonic_attention_process_train(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Calculating monotonic attention process for training
        Including:
            stepwise probability: p_choose
            expected hard alignment: alpha
            expected soft attention: beta
        """
        assert query is not None
        assert key is not None

        # 1. compute stepwise probability
        p_choose = self.p_choose_from_qk(query, key, key_padding_mask)

        # 2. compute expected_alignment
        alpha = expected_alignment_from_p_choose(
            p_choose,
            key_padding_mask,
            eps=self.eps,
        )

        if self.mass_preservation:
            alpha = mass_preservation(
                alpha, key_padding_mask
            )

        # 3. compute expected soft attention (soft aligned model only)
        if self.soft_attention:
            soft_energy = self.energy_from_qk(
                query,
                key,
                "soft",
                key_padding_mask=None,
            )

            beta = expected_soft_attention(
                alpha,
                soft_energy,
                padding_mask=key_padding_mask,
                chunk_size=self.chunk_size,
                eps=self.eps,
            )
        else:
            beta = alpha
            soft_energy = alpha

        return p_choose, alpha, beta, soft_energy

    def forward(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True, static_kv: bool = False, need_head_weights: bool = False,
    ):
        """
        query: tgt_len, bsz, embed_dim
        key: src_len, bsz, embed_dim
        value: src_len, bsz, embed_dim
        """

        assert attn_mask is None
        assert query is not None
        assert key is not None
        assert value is not None

        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)

        if key_padding_mask is not None:
            assert not key_padding_mask[:, 0].any(), (
                "Only right padding is supported."
            )
            key_padding_mask = (
                key_padding_mask
                .unsqueeze(1)
                .expand([bsz, self.num_heads, src_len])
                .contiguous()
                .view(-1, src_len)
            )

        if incremental_state is not None:
            # Inference
            (
                p_choose, alpha, beta
            ) = self.monotonic_attention_process_infer(
                query, key, incremental_state
            )
            soft_energy = beta
        else:
            # Train
            (
                p_choose, alpha, beta, soft_energy
            ) = self.monotonic_attention_process_train(
                query, key, key_padding_mask
            )

        v = self.v_proj(value)
        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn = torch.bmm(beta.type_as(v), v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)

        p_choose = p_choose.view(bsz, self.num_heads, tgt_len, src_len)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)

        return attn, {
            "p_choose": p_choose,
            "alpha": alpha,
            "beta": beta,
        }

    def _get_monotonic_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        maybe_incremental_state = self.get_incremental_state(
            incremental_state,
            'monotonic',
        )
        if maybe_incremental_state is None:
            typed_empty_dict: Dict[str, Optional[Tensor]] = {}
            return typed_empty_dict
        else:
            return maybe_incremental_state

    def _set_monotonic_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]], buffer: Dict[str, Optional[Tensor]]):
        self.set_incremental_state(
            incremental_state,
            'monotonic',
            buffer,
        )


@register_monotonic_attention("infinite_lookback")
class MonotonicInfiniteLookbackAttention(
    MonotonicAttention
):
    def __init__(self, args):
        super().__init__(args)
        self.soft_attention = True
        self.init_soft_attention()

    def init_soft_attention(self):
        self.k_proj_soft = nn.Linear(self.kdim, self.embed_dim, bias=True)
        self.q_proj_soft = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_in_proj["soft"] = self.k_proj_soft
        self.q_in_proj["soft"] = self.q_proj_soft

        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.k_in_proj["soft"].weight, gain=1 / math.sqrt(2)
            )
            nn.init.xavier_uniform_(
                self.q_in_proj["soft"].weight, gain=1 / math.sqrt(2)
            )
        else:
            nn.init.xavier_uniform_(self.k_in_proj["soft"].weight)
            nn.init.xavier_uniform_(self.q_in_proj["soft"].weight)


@register_monotonic_attention("waitk")
class WaitKAttention(
    MonotonicInfiniteLookbackAttention
):
    """
    STACL: Simultaneous Translation with Implicit Anticipation and
    Controllable Latency using Prefix-to-Prefix Framework
    https://www.aclweb.org/anthology/P19-1289/
    """
    def __init__(self, args):
        super().__init__(args)
        self.q_in_proj["soft"] = self.q_in_proj["monotonic"]
        self.k_in_proj["soft"] = self.k_in_proj["monotonic"]

        self.waitk_lagging = args.waitk_lagging
        assert self.waitk_lagging > 0, (
            f"Lagging has to been larger than 0, get {self.waitk_lagging}."
        )

    @staticmethod
    def add_args(parser):
        super(
            MonotonicInfiniteLookbackAttention,
            MonotonicInfiniteLookbackAttention
        ).add_args(parser)

        parser.add_argument(
            "--waitk-lagging", type=int, required=True, help="Wait K lagging"
        )

    def p_choose_from_qk(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        assert query is not None
        assert key is not None

        p_choose = waitk_p_choose(
            tgt_len=query.size(0),
            src_len=key.size(0),
            bsz=query.size(1) * self.num_heads,
            waitk_lagging=self.waitk_lagging,
            key_padding_mask=key_padding_mask,
            incremental_state=incremental_state,
        )

        return p_choose.to(query)


@register_monotonic_attention("chunkwise")
class ChunkwiseAttention(
    MonotonicInfiniteLookbackAttention
):
    def __init__(self, args):
        super().__init__(args)
        self.chunk_size = args.mocha_chunk_size
        assert self.chunk_size > 1

    @staticmethod
    def add_args(parser):
        super(
            MonotonicInfiniteLookbackAttention
        ).add_args(parser)

        parser.add_argument(
            "--mocha-chunk-size", type=int,
            required=True, help="Mocha chunk size"
        )
