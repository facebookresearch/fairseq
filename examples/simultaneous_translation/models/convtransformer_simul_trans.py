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
    convtransformer_espnet,
    ConvTransformerEncoder,
)
from fairseq.models.speech_to_text.modules.augmented_memory_attention import (
    augmented_memory,
    SequenceEncoder,
    AugmentedMemoryConvTransformerEncoder,
)
from fairseq.models.speech_to_text.modules.emformer import emformer_encoder


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


@emformer_encoder
class ConvTransformerEmformerEncoder(ConvTransformerEncoder):
    pass


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
