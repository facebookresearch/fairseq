# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules.transformer_layer import TransformerEncoderLayer, TransformerDecoderLayer
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class Adapter(nn.Module):
    def __init__(self, cfg, red_fac=2):
        super(Adapter, self).__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder_embed_dim
        self.quant_noise = getattr(cfg, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(cfg, "quant_noise_pq_block_size", 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(cfg, "activation_fn", "relu") or "relu"
        )
        self.fc1 = quant_noise(
            nn.Linear(self.embed_dim, self.embed_dim // red_fac),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc2 = quant_noise(
            nn.Linear(self.embed_dim // red_fac, self.embed_dim),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        activation_dropout_p = getattr(cfg, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = getattr(cfg, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        if not hasattr(self.cfg, "adapter_dropout") or self.cfg.adapter_dropout:
            x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class XMODTransformerEncoderLayerBase(TransformerEncoderLayer):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        if hasattr(cfg, "adapter_modules") and cfg.adapter_modules:
            export = getattr(cfg, "export", False)
            if cfg.adapter_layer_norm:
                self.adapter_layer_norm = LayerNorm(self.embed_dim, export=export)
            self.adapter_modules = nn.ModuleDict(dict())
            if hasattr(self.cfg, "bottleneck"):
                bottleneck = self.cfg.bottleneck
            else:
                bottleneck = 2
            for language in cfg.languages:
                self.adapter_modules[str(language)] = Adapter(cfg, red_fac=bottleneck)

    def lang_adapter(self, lang_id, x):
        # If language adapters exist pass throught them
        if hasattr(self.cfg, "adapter_modules") and self.cfg.adapter_modules:
            if lang_id is None:
                lang_id = ["en_XX"] * x.shape[1]
            d_langs = [lang_id[0]]
            lang_lengths = [1]
            for lang in lang_id[1:]:
                if lang == d_langs[-1]:
                    lang_lengths[-1] += 1
                else:
                    d_langs.append(lang)
                    lang_lengths.append(1)

            if (
                not hasattr(self.cfg, "ln_before_adapter")
                or not self.cfg.ln_before_adapter
            ):
                residual = x
            if self.cfg.adapter_layer_norm:
                x = self.adapter_layer_norm(x)
            elif self.cfg.adapter_reuse_layer_norm:
                x = self.final_layer_norm(x)
            if hasattr(self.cfg, "ln_before_adapter") and self.cfg.ln_before_adapter:
                residual = x

            split_x = torch.split(x, lang_lengths, 1)
            x_ = []
            for i, (lang, s_x) in enumerate(zip(d_langs, split_x)):
                lang = lang.replace("_rom", "").replace("_zaw", "")
                x_.append(self.adapter_modules[str(lang)](s_x))
            x = torch.cat(x_, 1)

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)

        return x

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        lang_id: Optional[list] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)   
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        x = self.lang_adapter(lang_id, x)

        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x



class XMODTransformerDecoderLayerBase(TransformerDecoderLayer):
    """Decoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg, no_encoder_attn=False):
        super().__init__(cfg, no_encoder_attn)
        if hasattr(cfg, "adapter_modules") and cfg.adapter_modules:
            export = getattr(cfg, "export", False)
            if cfg.adapter_layer_norm:
                self.adapter_layer_norm = LayerNorm(self.embed_dim, export=export)
            self.adapter_modules = nn.ModuleDict(dict())
            if hasattr(self.cfg, "bottleneck"):
                bottleneck = self.cfg.bottleneck
            else:
                bottleneck = 2
            for language in cfg.languages:
                self.adapter_modules[str(language)] = Adapter(cfg, red_fac=bottleneck)

    def lang_adapter(self, lang_id, x):
        # If language adapters exist pass throught them
        if hasattr(self.cfg, "adapter_modules") and self.cfg.adapter_modules:
            if lang_id is None:
                lang_id = ["en_XX"] * x.shape[1]
            d_langs = [lang_id[0]]
            lang_lengths = [1]
            for lang in lang_id[1:]:
                if lang == d_langs[-1]:
                    lang_lengths[-1] += 1
                else:
                    d_langs.append(lang)
                    lang_lengths.append(1)

            if (
                not hasattr(self.cfg, "ln_before_adapter")
                or not self.cfg.ln_before_adapter
            ):
                residual = x
            if self.cfg.adapter_layer_norm:
                x = self.adapter_layer_norm(x)
            elif self.cfg.adapter_reuse_layer_norm:
                x = self.final_layer_norm(x)
            if hasattr(self.cfg, "ln_before_adapter") and self.cfg.ln_before_adapter:
                residual = x

            split_x = torch.split(x, lang_lengths, 1)
            x_ = []
            for i, (lang, s_x) in enumerate(zip(d_langs, split_x)):
                lang = lang.replace("_rom", "").replace("_zaw", "")
                x_.append(self.adapter_modules[str(lang)](s_x))
            x = torch.cat(x_, 1)

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)

        return x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        lang_id: Optional[list] = None,
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
                incremental_state=incremental_state,
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
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)

        x = self.lang_adapter(lang_id, x)

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
            return x, attn, self_attn_state
        return x, attn, None
