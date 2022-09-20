# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from pathlib import Path

import torch

from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from fairseq.models.speech_to_text.modules.convolution import (
    Conv1dSubsampler,
    Conv2dSubsampler,
)
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder,
    S2TTransformerModel,
)
from fairseq.models.speech_to_text.s2t_transformer import (
    base_architecture as transformer_base_architecture,
)
from fairseq.modules import PositionalEmbedding, RelPositionalEncoding
from fairseq.modules.conformer_layer import ConformerEncoderLayer

logger = logging.getLogger(__name__)


class S2TConformerEncoder(FairseqEncoder):
    """Conformer Encoder for speech translation based on https://arxiv.org/abs/2005.08100"""

    def __init__(self, args):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1
        self.conv_version = args.conv_version
        if self.conv_version == "s2t_transformer":
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
        elif self.conv_version == "convtransformer":
            self.subsample = Conv2dSubsampler(
                args.input_channels,
                args.input_feat_per_channel,
                args.conv_out_channels,
                args.encoder_embed_dim,
            )
        self.pos_enc_type = args.pos_enc_type
        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                args.max_source_positions, args.encoder_embed_dim
            )
        elif self.pos_enc_type == "rope":
            self.embed_positions = None
        else:  # Use absolute positional embedding
            self.pos_enc_type = "abs"
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.linear = torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.dropout = torch.nn.Dropout(args.dropout)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    embed_dim=args.encoder_embed_dim,
                    ffn_embed_dim=args.encoder_ffn_embed_dim,
                    attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                    attn_type=args.attn_type,
                    pos_enc_type=self.pos_enc_type,
                    use_fp16=args.fp16,
                )
                for _ in range(args.encoder_layers)
            ]
        )

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        """
        Args:
            src_tokens: Input source tokens Tensor of shape B X T X C
            src_lengths: Lengths Tensor corresponding to input source tokens
            return_all_hiddens: If true will append the self attention states to the encoder states
        Returns:
            encoder_out: Tensor of shape B X T X C
            encoder_padding_mask: Optional Tensor with mask
            encoder_embedding: Optional Tensor. Always empty here
            encoder_states: List of Optional Tensors wih self attention states
            src_tokens: Optional Tensor. Always empty here
            src_lengths: Optional Tensor. Always empty here
        """
        x, input_lengths = self.subsample(src_tokens, src_lengths)  # returns T X B X C
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        x = self.embed_scale * x
        if self.pos_enc_type == "rel_pos":
            positions = self.embed_positions(x)

        elif self.pos_enc_type == "rope":
            positions = None

        else:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            positions = None

        x = self.linear(x)
        x = self.dropout(x)
        encoder_states = []

        # x is T X B X C
        for layer in self.conformer_layers:
            x, _ = layer(x, encoder_padding_mask, positions)
            if return_all_hiddens:
                encoder_states.append(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens,
                    src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
        else:
            x = self._forward(
                src_tokens,
                src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        """Required method for a FairseqEncoder. Calls the method from the parent class"""
        return S2TTransformerEncoder.reorder_encoder_out(self, encoder_out, new_order)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


@register_model("s2t_conformer")
class S2TConformerModel(S2TTransformerModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="dimension of input features per channel",
        )
        parser.add_argument(
            "--input-channels",
            type=int,
            metavar="N",
            help="number of chennels of input features",
        )
        parser.add_argument(
            "--depthwise-conv-kernel-size",
            type=int,
            metavar="N",
            help="kernel size of depthwise convolution layers",
        )
        parser.add_argument(
            "--attn-type",
            type=str,
            metavar="STR",
            help="If not specified uses fairseq MHA. Other valid option is espnet",
        )
        parser.add_argument(
            "--pos-enc-type",
            type=str,
            metavar="STR",
            help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TConformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder


@register_model_architecture("s2t_conformer", "s2t_conformer")
def conformer_base_architecture(args):
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.input_channels = getattr(args, "input_channels", 1)
    args.max_source_positions = getattr(args, "max_source_positions", 6000)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    transformer_base_architecture(args)
