# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules.transformer_layer import TransformerEncoderLayer
from typing import Optional
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
