# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import (
    ConvTransformerModel,
    ConvTransformerEncoder,
)
from fairseq.models.speech_to_text.modules.augmented_memory_attention import (
    augmented_memory,
    SequenceEncoder,
    AugmentedMemoryConvTransformerEncoder,
)

from torch import nn, Tensor
from typing import Dict, List
from fairseq.models.speech_to_text.modules.emformer import NoSegAugmentedMemoryTransformerEncoderLayer

@register_model("convtransformer_simul_trans")
class SimulConvTransformerModel(ConvTransformerModel):
    """
    Implementation of the paper:

    SimulMT to SimulST: Adapting Simultaneous Text Translation to
    End-to-End Simultaneous Speech Translation

    https://www.aclweb.org/anthology/2020.aacl-main.58.pdf
    """

    @staticmethod
    def add_args(parser):
        super(SimulConvTransformerModel, SimulConvTransformerModel).add_args(parser)
        parser.add_argument(
            "--train-monotonic-only",
            action="store_true",
            default=False,
            help="Only train monotonic attention",
        )

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.tgt_dict

        from examples.simultaneous_translation.models.transformer_monotonic_attention import (
            TransformerMonotonicDecoder,
        )

        decoder = TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder


@register_model_architecture(
    "convtransformer_simul_trans", "convtransformer_simul_trans_espnet"
)
def convtransformer_simul_trans_espnet(args):
    convtransformer_espnet(args)


@register_model("convtransformer_augmented_memory")
@augmented_memory
class AugmentedMemoryConvTransformerModel(SimulConvTransformerModel):
    @classmethod
    def build_encoder(cls, args):
        encoder = SequenceEncoder(args, AugmentedMemoryConvTransformerEncoder(args))

        if getattr(args, "load_pretrained_encoder_from", None) is not None:
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )

        return encoder


@register_model_architecture(
    "convtransformer_augmented_memory", "convtransformer_augmented_memory"
)
def augmented_memory_convtransformer_espnet(args):
    convtransformer_espnet(args)


# ============================================================================ #
#   Convtransformer
#   with monotonic attention decoder
#   with emformer encoder
# ============================================================================ #


class ConvTransformerEmformerEncoder(ConvTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)
        stride = self.conv_layer_stride(args)
        trf_left_context = args.segment_left_context // stride
        trf_right_context = args.segment_right_context // stride
        context_config = [trf_left_context, trf_right_context]
        self.transformer_layers = nn.ModuleList(
            [
                NoSegAugmentedMemoryTransformerEncoderLayer(
                    input_dim=args.encoder_embed_dim,
                    num_heads=args.encoder_attention_heads,
                    ffn_dim=args.encoder_ffn_embed_dim,
                    num_layers=args.encoder_layers,
                    dropout_in_attn=args.dropout,
                    dropout_on_attn=args.dropout,
                    dropout_on_fc1=args.dropout,
                    dropout_on_fc2=args.dropout,
                    activation_fn=args.activation_fn,
                    context_config=context_config,
                    segment_size=args.segment_length,
                    max_memory_size=args.max_memory_size,
                    scaled_init=True,  # TODO: use constant for now.
                    tanh_on_mem=args.amtrf_tanh_on_mem,
                )
            ]
        )
        self.conv_transformer_encoder = ConvTransformerEncoder(args)

    def forward(self, src_tokens, src_lengths):
        encoder_out: Dict[str, List[Tensor]] = self.conv_transformer_encoder(src_tokens, src_lengths.to(src_tokens.device))
        output = encoder_out["encoder_out"][0]
        encoder_padding_masks = encoder_out["encoder_padding_mask"]

        return {
            "encoder_out": [output],
            # This is because that in the original implementation
            # the output didn't consider the last segment as right context.
            "encoder_padding_mask": [encoder_padding_masks[0][:, : output.size(0)]] if len(encoder_padding_masks) > 0
            else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

    @staticmethod
    def conv_layer_stride(args):
        # TODO: make it configurable from the args
        return 4


@register_model("convtransformer_emformer")
class ConvtransformerEmformer(SimulConvTransformerModel):
    @staticmethod
    def add_args(parser):
        super(ConvtransformerEmformer, ConvtransformerEmformer).add_args(parser)

        parser.add_argument(
            "--segment-length",
            type=int,
            metavar="N",
            help="length of each segment (not including left context / right context)",
        )
        parser.add_argument(
            "--segment-left-context",
            type=int,
            help="length of left context in a segment",
        )
        parser.add_argument(
            "--segment-right-context",
            type=int,
            help="length of right context in a segment",
        )
        parser.add_argument(
            "--max-memory-size",
            type=int,
            default=-1,
            help="Right context for the segment.",
        )
        parser.add_argument(
            "--amtrf-tanh-on-mem",
            default=False,
            action="store_true",
            help="whether to use tanh on memory vector",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = ConvTransformerEmformerEncoder(args)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder


@register_model_architecture(
    "convtransformer_emformer",
    "convtransformer_emformer",
)
def convtransformer_emformer_base(args):
    convtransformer_espnet(args)


def convtransformer_espnet(args):
    # convtransformer / convtransformer_espnet
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    # convtransformer / base_architecture
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.max_source_positions = getattr(args, "max_source_positions", 3000)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)
