# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from numpy.random import uniform
from torch import Tensor

from fairseq.modules import LayerNorm
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase


class AugTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    """Decoder layer block augmented with an additional cross-attention.

    This decoder block is processed with the sequence of the following sub-modules.
        self-attention -> cross-attention (first) -> cross-attention (second) -> FFN

    Args:
        cfg (argparse.Namespace): parsed command-line arguments
        encoder_attn_merge_type (str, optional): the way to combine outputs from
            two cross-attention modules. If "sequential" is set, two cross-attention
            modules are stacked sequentially. If "parallel" is set, they are processed
            in parallel and combined before feeding it to FFN (default: sequential).
        dropnet_ratio (float, optional): a probability to drop each cross-attention
            module during training (default: 0.0).
    """

    def __init__(
        self,
        cfg,
        add_bias_kv=False,
        add_zero_attn=False,
        encoder_attn_merge_type="sequential",
        dropnet_ratio=0.0,
    ):
        super().__init__(
            cfg,
            no_encoder_attn=False,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
        )
        self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.encoder_attn2 = self.build_encoder_attention(self.embed_dim, cfg)
        if encoder_attn_merge_type == "sequential":
            self.encoder_attn_layer_norm2 = LayerNorm(self.embed_dim, export=cfg.export)
        else:
            self.encoder_attn_layer_norm2 = None

        self.encoder_attn_merge_type = encoder_attn_merge_type
        self.dropnet_ratio = dropnet_ratio

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        encoder_out_aug: Optional[torch.Tensor] = None,
        encoder_padding_mask2: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        assert encoder_out is not None
        assert encoder_out_aug is not None

        if self.encoder_attn_merge_type == "sequential":
            ratios = self.get_dropnet_ratio()

            # first encoder attention
            if ratios[0] > 0:
                residual = x
                if self.normalize_before:
                    x = self.encoder_attn_layer_norm(x)
                if prev_attn_state is not None:
                    prev_key, prev_value = prev_attn_state[:2]
                    saved_state: Dict[str, Optional[Tensor]] = {
                        "prev_key": prev_key,
                        "prev_value": prev_value,
                    }
                    if len(prev_attn_state) >= 3:
                        saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                    assert incremental_state is not None
                    self.encoder_attn._set_input_buffer(incremental_state, saved_state)

                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                )
                x = self.dropout_module(x)
                x = self.residual_connection(x, residual)
                if not self.normalize_before:
                    x = self.encoder_attn_layer_norm(x)
                x = ratios[0] * x

            # second encoder attention
            if ratios[1] > 0:
                residual = x
                if self.normalize_before:
                    x = self.encoder_attn_layer_norm2(x)
                if prev_attn_state is not None:
                    prev_key, prev_value = prev_attn_state[:2]
                    saved_state: Dict[str, Optional[Tensor]] = {
                        "prev_key": prev_key,
                        "prev_value": prev_value,
                    }
                    if len(prev_attn_state) >= 3:
                        saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                    assert incremental_state is not None
                    self.encoder_attn2._set_input_buffer(incremental_state, saved_state)

                x, attn2 = self.encoder_attn2(
                    query=x,
                    key=encoder_out_aug,
                    value=encoder_out_aug,
                    key_padding_mask=encoder_padding_mask2,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                )
                x = self.dropout_module(x)
                x = self.residual_connection(x, residual)
                if not self.normalize_before:
                    x = self.encoder_attn_layer_norm2(x)
                x = ratios[1] * x

        elif self.encoder_attn_merge_type == "parallel":
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x1, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x2, attn2 = self.encoder_attn2(
                query=x,
                key=encoder_out_aug,
                value=encoder_out_aug,
                key_padding_mask=encoder_padding_mask2,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x1 = self.dropout_module(x1)
            x2 = self.dropout_module(x2)
            ratios = self.get_dropnet_ratio()
            x = ratios[0] * x1 + ratios[1] * x2
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        else:
            raise NotImplementedError(self.encoder_attn_merge_type)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, attn2, self_attn_state
        return x, attn, attn2, None

    def get_dropnet_ratio(self):
        if self.encoder_attn_merge_type == "sequential":
            if self.dropnet_ratio > 0:
                frand = float(uniform(0, 1))
                if frand < self.dropnet_ratio and self.training:
                    return [2, 0]
                elif frand > 1 - self.dropnet_ratio and self.training:
                    return [0, 2]
                else:
                    return [1, 1]
            else:
                return [1, 1]

        elif self.encoder_attn_merge_type == "parallel":
            if self.dropnet_ratio > 0:
                frand = float(uniform(0, 1))
                if frand < self.dropnet_ratio and self.training:
                    return [1, 0]
                elif frand > 1 - self.dropnet_ratio and self.training:
                    return [0, 1]
                else:
                    return [0.5, 0.5]
            else:
                return [0.5, 0.5]
