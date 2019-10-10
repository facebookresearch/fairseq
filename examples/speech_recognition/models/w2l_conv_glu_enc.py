#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)


default_conv_enc_config = """[ 
    (400, 13, 170, 0.2),
    (440, 14, 0, 0.214),
    (484, 15, 0, 0.22898),
    (532, 16, 0, 0.2450086),
    (584, 17, 0, 0.262159202),
    (642, 18, 0, 0.28051034614),
    (706, 19, 0, 0.30014607037),
    (776, 20, 0, 0.321156295296),
    (852, 21, 0, 0.343637235966),
    (936, 22, 0, 0.367691842484),
    (1028, 23, 0, 0.393430271458),
    (1130, 24, 0, 0.42097039046),
    (1242, 25, 0, 0.450438317792),
    (1366, 26, 0, 0.481969000038),
    (1502, 27, 0, 0.51570683004),
    (1652, 28, 0, 0.551806308143),
    (1816, 29, 0, 0.590432749713),
]"""


@register_model("asr_w2l_conv_glu_encoder")
class W2lConvGluEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder):
        super().__init__(encoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--in-channels",
            type=int,
            metavar="N",
            help="number of encoder input channels",
        )
        parser.add_argument(
            "--conv-enc-config",
            type=str,
            metavar="EXPR",
            help="""
    an array of tuples each containing the configuration of one conv layer
    [(out_channels, kernel_size, padding, dropout), ...]
            """,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        conv_enc_config = getattr(args, "conv_enc_config", default_conv_enc_config)
        encoder = W2lConvGluEncoder(
            vocab_size=len(task.target_dictionary),
            input_feat_per_channel=args.input_feat_per_channel,
            in_channels=args.in_channels,
            conv_enc_config=eval(conv_enc_config),
        )
        return cls(encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = False
        return lprobs


class W2lConvGluEncoder(FairseqEncoder):
    def __init__(
        self, vocab_size, input_feat_per_channel, in_channels, conv_enc_config
    ):
        super().__init__(None)

        self.input_dim = input_feat_per_channel
        if in_channels != 1:
            raise ValueError("only 1 input channel is currently supported")

        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropouts = []
        cur_channels = input_feat_per_channel

        for out_channels, kernel_size, padding, dropout in conv_enc_config:
            layer = nn.Conv1d(cur_channels, out_channels, kernel_size, padding=padding)
            layer.weight.data.mul_(math.sqrt(3))  # match wav2letter init
            self.conv_layers.append(nn.utils.weight_norm(layer))
            self.dropouts.append(dropout)
            if out_channels % 2 != 0:
                raise ValueError("odd # of out_channels is incompatible with GLU")
            cur_channels = out_channels // 2  # halved by GLU

        for out_channels in [2 * cur_channels, vocab_size]:
            layer = nn.Linear(cur_channels, out_channels)
            layer.weight.data.mul_(math.sqrt(3))
            self.linear_layers.append(nn.utils.weight_norm(layer))
            cur_channels = out_channels // 2

    def forward(self, src_tokens, src_lengths, **kwargs):

        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """
        B, T, _ = src_tokens.size()
        x = src_tokens.transpose(1, 2).contiguous()  # (B, feat, T) assuming C == 1

        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)
            x = F.glu(x, dim=1)
            x = F.dropout(x, p=self.dropouts[layer_idx], training=self.training)

        x = x.transpose(1, 2).contiguous()  # (B, T, 908)
        x = self.linear_layers[0](x)
        x = F.glu(x, dim=2)
        x = F.dropout(x, p=self.dropouts[-1])
        x = self.linear_layers[1](x)

        assert x.size(0) == B
        assert x.size(1) == T

        encoder_out = x.transpose(0, 1)  # (T, B, vocab_size)

        # need to debug this -- find a simpler/elegant way in pytorch APIs
        encoder_padding_mask = (
            torch.arange(T).view(1, T).expand(B, -1).to(x.device)
            >= src_lengths.view(B, 1).expand(-1, T)
        ).t()  # (B x T) -> (T x B)

        return {
            "encoder_out": encoder_out,  # (T, B, vocab_size)
            "encoder_padding_mask": encoder_padding_mask,  # (T, B)
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )
        encoder_out["encoder_padding_mask"] = encoder_out[
            "encoder_padding_mask"
        ].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (1e6, 1e6)  # an arbitrary large number


@register_model_architecture("asr_w2l_conv_glu_encoder", "w2l_conv_glu_enc")
def w2l_conv_glu_enc(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.in_channels = getattr(args, "in_channels", 1)
    args.conv_enc_config = getattr(args, "conv_enc_config", default_conv_enc_config)
