# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from fairseq.models import FairseqEncoderModel, register_model, register_model_architecture
from fairseq.models.speech_to_text.vggtransformer import (
    VGGTransformerEncoder, DEFAULT_ENC_VGGBLOCK_CONFIG, DEFAULT_ENC_TRANSFORMER_CONFIG
)


@register_model("vggtransformer_encoder")
class VGGTransformerEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder):
        super().__init__(encoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--input-feat-per-channel", type=int, metavar="N",
                            help="encoder input dimension per input channel")
        parser.add_argument("--vggblock-enc-config", type=str, metavar="EXPR",
                            help="an array of tuples each containing the configuration of one vggblock "
                                 "[(out_channels, conv_kernel_size, pooling_kernel_size,num_conv_layers), ...]")
        parser.add_argument("--transformer-enc-config", type=str, metavar="EXPR",
                            help="a tuple containing the configuration of the Transformer layers configurations: "
                                 "[(input_dim, num_heads, ffn_dim, normalize_before, dropout, attention_dropout, relu_dropout), ]")
        parser.add_argument("--enc-output-dim", type=int, metavar="N",
                            help="encoder output dimension, projecting the LSTM output")
        parser.add_argument("--in-channels", type=int,metavar="N",
                            help="number of encoder input channels")
        parser.add_argument("--transformer-context", type=str, metavar="EXPR",
                            help="either None or a tuple of two ints, indicating left/right context a transformer can have access to")
        parser.add_argument("--transformer-sampling", type=str, metavar="EXPR",
                            help="either None or a tuple of ints, indicating sampling factor in each layer")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture_enc_only(args)
        encoder = VGGTransformerEncoderOnly(
            vocab_size=len(task.target_dictionary),
            input_feat_per_channel=args.input_feat_per_channel,
            vggblock_config=eval(args.vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.enc_output_dim,
            in_channels=args.in_channels,
            transformer_context=eval(args.transformer_context),
            transformer_sampling=eval(args.transformer_sampling),
        )
        return cls(encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (T, B, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        # lprobs is a (T, B, D) tensor
        # we need to transoose to get (B, T, D) tensor
        lprobs = lprobs.transpose(0, 1).contiguous()
        lprobs.batch_first = True
        return lprobs


class VGGTransformerEncoderOnly(VGGTransformerEncoder):
    def __init__(
        self,
        vocab_size,
        input_feat_per_channel,
        vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
        transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
        encoder_output_dim=512,
        in_channels=1,
        transformer_context=None,
        transformer_sampling=None,
    ):
        super().__init__(
            input_feat_per_channel=input_feat_per_channel,
            vggblock_config=vggblock_config,
            transformer_config=transformer_config,
            encoder_output_dim=encoder_output_dim,
            in_channels=in_channels,
            transformer_context=transformer_context,
            transformer_sampling=transformer_sampling,
        )
        self.fc_out = nn.Linear(self.encoder_output_dim, vocab_size)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """

        enc_out = super().forward(src_tokens, src_lengths)
        x = self.fc_out(enc_out["encoder_out"])
        # x = F.log_softmax(x, dim=-1)
        # Note: no need this line, because model.get_normalized_prob will call
        # log_softmax
        return {
            "encoder_out": x,  # (T, B, C)
            "encoder_padding_mask": enc_out["encoder_padding_mask"],  # (T, B)
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1e6, 1e6  # an arbitrary large number


def base_architecture_enc_only(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 40)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(32, 3, 2, 2, True)] * 2"
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", "((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2"
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.in_channels = getattr(args, "in_channels", 1)
    args.transformer_context = getattr(args, "transformer_context", "None")
    args.transformer_sampling = getattr(args, "transformer_sampling", "None")


@register_model_architecture("vggtransformer_encoder", "vggtransformer_enc_1")
def vggtransformer_enc_1(args):
    # vggtransformer_1 is the same as vggtransformer_enc_big, except the number
    # of layers is increased to 16
    # keep it here for backward compatiablity purpose
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args,
        "transformer_enc_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16",
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
