# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import json
import torch
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils

# from rotary_embedding_torch import RotaryEmbedding
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.multihead_attention import MultiheadAttention

from fairseq.modules.rotary_positional_embedding import (
    apply_rotary_pos_emb,
    RotaryPositionalEmbedding,
    LinearScalingRotaryPositionalEmbedding,
    DynamicNTKScalingRotaryPositionalEmbedding,
    YaRNScaledRotaryPositionalEmbedding,
    DynamicYaRNScaledRotaryPositionalEmbedding,
)


class NativeMultiheadAttention(MultiheadAttention):
    """Native Multi-headed attention
    Removes a lot of the overhead in the MultiheadAttention module
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        dictionary=None,
        q_noise=0.0,
        qn_block_size=8,
        rope_args=None,
        yarn_args=None,
    ):
        super().__init__(embed_dim, num_heads, dictionary=dictionary)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.rope_args = json.loads(rope_args) if rope_args is not None else None
        self.yarn_args = json.loads(yarn_args) if yarn_args is not None else None
        self.yarn_pos_embed = None
        self.rotary_pos_embed = None

        # both self.rope_args and self.yarn_args cannot be set at the same time
        assert not (
            self.rope_args is not None and self.yarn_args is not None
        ), "Both rotary and yarn position embeddings cannot be set at the same time"

        if self.rope_args is not None:
            if self.rope_args["type"] == "vanilla":
                self.rotary_pos_embed = RotaryPositionalEmbedding(
                    dim=self.head_dim,
                    base=self.rope_args.get("base", 10000),
                    max_position_embeddings=self.rope_args.get(
                        "max_position_embeddings", 2048
                    ),
                )
            elif self.rope_args["type"] == "linear":
                self.rotary_pos_embed = LinearScalingRotaryPositionalEmbedding(
                    dim=self.head_dim,
                    base=self.rope_args.get("base", 10000),
                    scaling_factor=self.rope_args.get("scaling_factor", 1.0),
                    max_position_embeddings=self.rope_args.get(
                        "max_position_embeddings", 2048
                    ),
                )
            elif self.rope_args["type"] == "dynamic":
                self.rotary_pos_embed = DynamicNTKScalingRotaryPositionalEmbedding(
                    dim=self.head_dim,
                    base=self.rope_args.get("base", 10000),
                    max_position_embeddings=self.rope_args.get(
                        "max_position_embeddings", 2048
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown rotary position embedding type: {self.rope_args['type']}. Allowed types are: vanilla, linear, dynamic"
                )

        if self.yarn_args is not None:
            if self.yarn_args["type"] == "vanilla":
                self.yarn_pos_embed = YaRNScaledRotaryPositionalEmbedding(
                    dim=self.head_dim,
                    base=self.yarn_args.get("base", 10000),
                    scale=self.yarn_args.get("scale", 1.0),
                    max_position_embeddings=self.yarn_args.get(
                        "max_position_embeddings", 2048
                    ),
                    original_max_position_embeddings=self.yarn_args.get(
                        "original_max_position_embeddings", 256
                    ),
                    extrapolation_factor=self.yarn_args.get(
                        "extrapolation_factor", 1.0
                    ),
                    attn_factor=self.yarn_args.get("attn_factor", 1),
                    beta_fast=self.yarn_args.get("beta_fast", 32),
                    beta_slow=self.yarn_args.get("beta_slow", 2),
                )
            elif self.yarn_args["type"] == "dynamic":
                self.yarn_pos_embed = DynamicYaRNScaledRotaryPositionalEmbedding(
                    dim=self.head_dim,
                    base=self.yarn_args.get("base", 10000),
                    max_position_embeddings=self.yarn_args.get(
                        "max_position_embeddings", 2048
                    ),
                    original_max_position_embeddings=self.yarn_args.get(
                        "original_max_position_embeddings", 256
                    ),
                    extrapolation_factor=self.yarn_args.get(
                        "extrapolation_factor", 1.0
                    ),
                    attn_factor=self.yarn_args.get("attn_factor", 1),
                    beta_fast=self.yarn_args.get("beta_fast", 32),
                    beta_slow=self.yarn_args.get("beta_slow", 2),
                    finetuned=self.yarn_args.get("finetuned", False),
                )
            else:
                raise ValueError(
                    f"Unknown rotary position embedding type: {self.yarn_args['type']}. Allowed types are: vanilla, dynamic"
                )

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.beam_size = 1
        self.reset_parameters()

        self.init_incremental_state()

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[
                        :, :, 0, :
                    ]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(
                            -1, self.beam_size, key_padding_mask.size(1)
                        )[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz
            )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = (
                k.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(
                    kv_bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = NativeMultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                kv_bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        if self.rotary_pos_embed is not None or self.yarn_pos_embed is not None:
            # q shape: [bsz * num_heads, tgt_len, head_dim]
            q_ = q.view(kv_bsz, self.num_heads, -1, self.head_dim)
            k_ = k.view(kv_bsz, self.num_heads, -1, self.head_dim)

            # this is mutually exclusive
            cos, sin = (
                self.rotary_pos_embed(q_, seq_len=q_.shape[2])
                if self.rotary_pos_embed is not None
                else self.yarn_pos_embed(q_, seq_len=q_.shape[2])
            )

            q_, k_ = apply_rotary_pos_emb(q_, k_, cos, sin)

            # reshape back to [bsz * num_heads, tgt_len, head_dim]
            q = q_.view(kv_bsz * self.num_heads, -1, self.head_dim)
            k = k_.view(kv_bsz * self.num_heads, -1, self.head_dim)

        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = torch.einsum(
                "bxhtd,bhsd->bxhts",
                q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]),
                k.view((kv_bsz, self.num_heads) + k.size()[1:]),
            )
            attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ], "attn_weights: {} vs [bsz * self.num_heads, tgt_len, src_len]: {}".format(
            list(attn_weights.size()), [bsz * self.num_heads, tgt_len, src_len]
        )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.view(
                    kv_bsz, -1, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=False)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn: Optional[Tensor] = None
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum(
                "bxhts,bhsd->bxhtd",
                attn_probs.view(
                    (
                        kv_bsz,
                        -1,
                        self.num_heads,
                    )
                    + attn_probs.size()[1:]
                ),
                v.view(
                    (
                        kv_bsz,
                        self.num_heads,
                    )
                    + v.size()[1:]
                ),
            )
            attn = attn.reshape((-1,) + attn.size()[-2:])
        else:
            attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights
