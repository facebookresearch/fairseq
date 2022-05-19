# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.utils import safe_getattr
from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer
from ..modules.multihead_attention_selection import MultiheadAttentionSelection


class HeadSelectionTransformerEncoderLayer(TransformerEncoderLayer):

    def __init__(self, args, layer_idx, attn_head_selector=None):
        super().__init__(args)
        self.layer_idx = layer_idx
        self.self_attn = self.build_self_attention_selection(
            self.embed_dim, args, attn_head_selector
        )

    def build_self_attention_selection(self, embed_dim, args, attn_head_selector=None):
        return MultiheadAttentionSelection(
            embed_dim,
            args.total_encoder_attention_heads,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            layer_idx=self.layer_idx,
            attn_head_selector=attn_head_selector
        )


class HeadSelectionTransformerDecoderLayer(TransformerDecoderLayer):

    def __init__(
        self,
        args,
        layer_idx,
        self_attn_head_selector=None,
        enc_attn_head_selector=None,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        self.layer_idx = layer_idx
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        if self_attn_head_selector is not None:
            self.self_attn = self.build_self_attention_selection(
                self.embed_dim, args,
                self_attn_head_selector=self_attn_head_selector,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn
            )
        if enc_attn_head_selector is not None:
            self.encoder_attn = self.build_encoder_attention_selection(
                self.embed_dim, args,
                enc_attn_head_selector=enc_attn_head_selector
            )

    def build_self_attention_selection(
        self, embed_dim, args, self_attn_head_selector=None,
        add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttentionSelection(
            embed_dim,
            args.total_decoder_attention_heads,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not safe_getattr(args, "cross_self_attention"),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            layer_idx=self.layer_idx,
            attn_head_selector=self_attn_head_selector,
        )

    def build_encoder_attention_selection(self, embed_dim, args, enc_attn_head_selector=None):
        return MultiheadAttentionSelection(
            embed_dim,
            args.total_decoder_attention_heads,
            args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            layer_idx=self.layer_idx,
            attn_head_selector=enc_attn_head_selector,
        )
