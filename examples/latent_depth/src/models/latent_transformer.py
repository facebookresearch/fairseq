# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch.nn as nn
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerDecoder, TransformerEncoder
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from torch import Tensor

from ..modules.latent_layers import LayerSelect


class LatentTransformerEncoder(TransformerEncoder):
    """Latent depth (https://arxiv.org/abs/2009.13102) implemented in
    TransformerEncoder.
    """

    def __init__(self, args, dictionary, embed_tokens, num_logits=1):
        self.num_logits = num_logits
        self.num_layers = args.encoder_layers
        super().__init__(args, dictionary, embed_tokens)
        self.layer_select = LayerSelect(
            num_layers=self.num_layers,
            num_logits=self.num_logits,
            soft_select=getattr(args, "soft_select", False),
            sampling_tau=getattr(args, "sampling_tau", 5.),
        )
        self.lang_idx = None
        self.layers = nn.ModuleList(
            [self._build_encoder_layer(args, idx) for idx in range(args.encoder_layers)]
        )

    def set_lang_idx(self, lang_idx):
        self.lang_idx = lang_idx

    def _build_encoder_layer(self, args, idx=None):
        return LatentTransformerEncoderLayer(args, idx, layer_select=self.layer_select)

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False):
        self.layer_select.sample(self.lang_idx)
        return super().forward(src_tokens, src_lengths, return_all_hiddens)


class LatentTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer with each (non_residual) block weighted by samples of Bernouli
    or Gumbel Signmoid samples.

    Args:
        args (argparse.Namespace): parsed command-line arguments from standard
            TransformerEncoderLayer.
        idx (int): layer index (used to retrieve samples).
        layer_select (LayerSelect, optional): instance of LayerSelect module with logits
            parameters and sampling method.
    """

    def __init__(self, args, idx, layer_select=None):
        super().__init__(args)
        self.idx = idx
        self.layer_select = layer_select

    def residual_connection(self, x, residual):
        return residual + x * self.layer_select(self.idx)


class LatentTransformerDecoder(TransformerDecoder):
    """Latent depth (https://arxiv.org/abs/2009.13102) implemented in
    TransformerDecoder.
    """

    def __init__(
        self, args, dictionary, embed_tokens, no_encoder_attn=False, num_logits=1
    ):
        self.num_logits = num_logits
        self.num_layers = args.decoder_layers
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.layer_select = LayerSelect(
            num_layers=self.num_layers,
            num_logits=self.num_logits,
            soft_select=getattr(args, "soft_select", False),
            sampling_tau=getattr(args, "sampling_tau", 5.),
        )
        self.lang_idx = None
        self.layers = nn.ModuleList(
            [
                self._build_decoder_layer(args, no_encoder_attn, idx)
                for idx in range(args.decoder_layers)
            ]
        )

    def set_lang_idx(self, lang_idx):
        self.lang_idx = lang_idx

    def _build_decoder_layer(self, args, no_encoder_attn=False, idx=None):
        return LatentTransformerDecoderLayer(
            args, idx, layer_select=self.layer_select, no_encoder_attn=no_encoder_attn
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        self.layer_select.sample(self.lang_idx)
        return super().forward(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            features_only=features_only,
            alignment_layer=alignment_layer,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )


class LatentTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer with each (non_residual) block weighted by samples of Bernouli
    or Gumbel Signmoid samples.

    Args:
        args (argparse.Namespace): parsed command-line arguments from standard
            TransformerDecoderLayer.
        idx (int): layer index (used to retrieve samples).
        layer_select (LayerSelect, optional): instance of LayerSelect module with logits
            parameters and sampling method.
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).

    """

    def __init__(
        self,
        args,
        idx,
        layer_select=None,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.idx = idx
        self.layer_select = layer_select

    def residual_connection(self, x, residual):
        return residual + x * self.layer_select(self.idx)
