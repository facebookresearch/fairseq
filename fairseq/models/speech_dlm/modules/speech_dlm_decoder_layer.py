# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class CrossChannelTransformerDecoderLayer(nn.Module):
    """Cross-Attention Transformer Decoder Layer block as described
    in the paper: https://arxiv.org/pdf/2203.16502.pdf

    Composed of a Multi-head Self Attention block followed by a
    Multi-head Cross-Attention block which attends to the self-attention
    outputs of the other channels. The weights of the attention blocks
    in all channels are shared.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        # This cross_self_attention is used for encoder-decoder systems,
        # It's not the cross-channel attention (defined below as cross_channel_attn)
        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.cross_channel_attn = self.build_cross_channel_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.cross_channel_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_cross_channel_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x_list_tensor: List[torch.Tensor],
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[
            List[Dict[str, Dict[str, Optional[Tensor]]]]
        ] = None,
        prev_self_attn_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x_list_tensor (List[Tensor]): list of input tensors in different channels,
                each tensor is of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            incremental_state (optional): list of incremental_state dictionaries over
                different channels (sequence generation mode)
            prev_self_attn_state (List[Tuple[Tensor, Tensor]], optional): list of tuples
                (self_attn_state, cross_channel_attn_state) over different channels
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            list of encoded output of shape `(seq_len, batch, embed_dim)`
        """
        n_channels = len(x_list_tensor)
        if need_head_weights:
            need_attn = True

        # incremental_state is a list of dictionaries over different channels
        if incremental_state is not None:
            assert isinstance(incremental_state, list)
            assert len(incremental_state) == n_channels

        # prev_self_attn_state is a list of tuples (self_attn_state, cross_channel_attn_state) over different channels
        if prev_self_attn_state is not None:
            assert isinstance(prev_self_attn_state, list)
            assert len(prev_self_attn_state) == n_channels
            for prev_self_attn_state_channel in prev_self_attn_state:
                assert isinstance(prev_self_attn_state_channel, tuple)
                assert len(prev_self_attn_state_channel) == 2

        # Backup for other channels & cross channel attention
        self_attn_mask_orin = self_attn_mask
        self_attn_padding_mask_orin = self_attn_padding_mask

        x_list = []
        attn_list = []
        for i, x in enumerate(x_list_tensor):
            residual = x

            if self.normalize_before:
                x = self.self_attn_layer_norm(x)

            if prev_self_attn_state is not None:
                prev_key, prev_value = prev_self_attn_state[i][0][:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_self_attn_state[i][0]) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[i][0][2]
                assert incremental_state is not None
                self.self_attn._set_input_buffer(incremental_state[i], saved_state)
            _self_attn_input_buffer = self.self_attn._get_input_buffer(
                incremental_state[i] if incremental_state is not None else None
            )
            if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
            ):
                if self_attn_mask_orin is not None:
                    assert encoder_out is not None
                    self_attn_mask = torch.cat(
                        (
                            x.new_zeros(x.size(0), encoder_out.size(0)),
                            self_attn_mask_orin,
                        ),
                        dim=1,
                    )
                if self_attn_padding_mask_orin is not None:
                    if encoder_padding_mask is None:
                        assert encoder_out is not None
                        encoder_padding_mask = self_attn_padding_mask_orin.new_zeros(
                            encoder_out.size(1), encoder_out.size(0)
                        )
                    self_attn_padding_mask = torch.cat(
                        (encoder_padding_mask, self_attn_padding_mask_orin), dim=1
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
                incremental_state=incremental_state[i]
                if incremental_state is not None
                else None,
                need_weights=False,
                attn_mask=self_attn_mask,
            )

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)

            if self.encoder_attn is not None and encoder_out is not None:
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
                    self.encoder_attn._set_input_buffer(
                        incremental_state[i], saved_state
                    )

                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state[i]
                    if incremental_state is not None
                    else None,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                )
                x = self.dropout_module(x)
                x = self.residual_connection(x, residual)
                if not self.normalize_before:
                    x = self.encoder_attn_layer_norm(x)

            x_list.append(x)
            attn_list.append(attn)

        # Store attentions & new x(s) (bc the old x(s) are used in other channels)
        x_list_new = []
        # Here comes the cross channel attention
        for i, x in enumerate(x_list):
            residual = x
            if self.normalize_before:
                x = self.cross_channel_attn_layer_norm(x)

            if prev_self_attn_state is not None:
                prev_key, prev_value = prev_self_attn_state[i][1][:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_self_attn_state[i][1]) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[i][1][2]
                assert incremental_state is not None
                self.cross_channel_attn._set_input_buffer(
                    incremental_state[i], saved_state
                )

            # The cross attention is computed with the concatenation of attentions from other channels
            if len(x_list) > 1:
                x_other = torch.cat(
                    [x_list[(i + j) % len(x_list)] for j in range(1, len(x_list))],
                    dim=0,
                )
            else:
                # Self-attention when having only one channel
                x_other = x_list[i]

            x, attn = self.cross_channel_attn(
                query=x,
                key=x_other,
                value=x_other,
                key_padding_mask=self_attn_padding_mask_orin,
                incremental_state=incremental_state[i]
                if incremental_state is not None
                else None,
                need_weights=False,
                attn_mask=self_attn_mask_orin,
            )

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.cross_channel_attn_layer_norm(x)

            x_list_new.append(x)
        x_list = x_list_new

        for i, x in enumerate(x_list):
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)

            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)

            x_list[i] = x
        # Trick for the checkpoint activation
        x_list_tensor = torch.stack(x_list)
        if self.onnx_trace and incremental_state is not None:
            self_and_cross_attn_state_list = []
            for i in range(n_channels):
                self_and_cross_attn_state = []
                for self_attn_module in [self.self_attn, self.cross_channel_attn]:
                    saved_state = self_attn_module._get_input_buffer(
                        incremental_state[i]
                    )
                    assert saved_state is not None
                    if self_attn_padding_mask is not None:
                        self_attn_module_state = [
                            saved_state["prev_key"],
                            saved_state["prev_value"],
                            saved_state["prev_key_padding_mask"],
                        ]
                    else:
                        self_attn_module_state = [
                            saved_state["prev_key"],
                            saved_state["prev_value"],
                        ]
                    self_and_cross_attn_state.append(self_attn_module_state)
                self_and_cross_attn_state_list.append(tuple(self_and_cross_attn_state))
            return x_list_tensor, attn_list, self_and_cross_attn_state_list
        return x_list_tensor, attn_list, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# Rewrite fairseq.modules.TransformerDecoderLayer
# to be compatible with checkpoint_activations
# (avoid forwarding model multiple times)
class StandardTransformerDecoderLayer(nn.Module):
    """Rewrite fairseq.modules.TransformerDecoderLayer to avoid forwarding
    model multiple times and be compatible with checkpoint_activations.

    The input is expected to be a list of tensors from different channels,
    each is forwarded to the same model (shared attention weights).

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x_list_tensor: List[torch.Tensor],
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[
            List[Dict[str, Dict[str, Optional[Tensor]]]]
        ] = None,
        prev_self_attn_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x_list_tensor (List[Tensor]): list of input tensors in different channels,
                each tensor is of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            incremental_state (optional): list of incremental_state dictionaries over
                different channels (sequence generation mode)
            prev_self_attn_state (List[Tuple[Tensor, Tensor]], optional): list of tuples
                (self_attn_state, cross_channel_attn_state) over different channels
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            list of encoded output of shape `(seq_len, batch, embed_dim)`
        """
        n_channels = len(x_list_tensor)
        if need_head_weights:
            need_attn = True

        # incremental_state is a list of dictionaries over different channels
        if incremental_state is not None:
            assert isinstance(incremental_state, list)
            assert len(incremental_state) == n_channels

        # prev_self_attn_state is a list of self_attn_state over different channels
        if prev_self_attn_state is not None:
            assert isinstance(prev_self_attn_state, list)
            assert len(prev_self_attn_state) == n_channels

        x_list = []
        attn_list = []
        for i, x in enumerate(x_list_tensor):
            residual = x

            if self.normalize_before:
                x = self.self_attn_layer_norm(x)

            if prev_self_attn_state is not None:
                prev_key, prev_value = prev_self_attn_state[i][:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_self_attn_state[i]) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
                assert incremental_state is not None
                self.self_attn._set_input_buffer(incremental_state[i], saved_state)
            _self_attn_input_buffer = self.self_attn._get_input_buffer(
                incremental_state
            )
            if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
            ):
                if self_attn_mask is not None:
                    assert encoder_out is not None
                    self_attn_mask = torch.cat(
                        (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask),
                        dim=1,
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
                incremental_state=incremental_state[i]
                if incremental_state is not None
                else None,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)

            if self.encoder_attn is not None and encoder_out is not None:
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
                    incremental_state=incremental_state[i]
                    if incremental_state is not None
                    else None,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                )
                x = self.dropout_module(x)
                x = self.residual_connection(x, residual)
                if not self.normalize_before:
                    x = self.encoder_attn_layer_norm(x)

            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)

            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)

            x_list.append(x)
            attn_list.append(attn)

        # Trick for the checkpoint activation
        x_list_tensor = torch.stack(x_list)
        if self.onnx_trace and incremental_state is not None:
            self_attn_state_list = []
            for i in range(n_channels):
                saved_state = self.self_attn._get_input_buffer(incremental_state[i])
                assert saved_state is not None
                if self_attn_padding_mask is not None:
                    self_attn_state = [
                        saved_state["prev_key"],
                        saved_state["prev_value"],
                        saved_state["prev_key_padding_mask"],
                    ]
                else:
                    self_attn_state = [
                        saved_state["prev_key"],
                        saved_state["prev_value"],
                    ]
                self_attn_state_list.append(self_attn_state)
            return x_list_tensor, attn_list, self_attn_state_list
        return x_list_tensor, attn_list, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
