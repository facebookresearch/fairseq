# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lightconv import Embedding, LightConvDecoder
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder


@register_model("lightconv_lm")
class LightConvLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            metavar="D",
            help="dropout probability",
        )
        parser.add_argument(
            "--attention-dropout",
            default=0.0,
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--relu-dropout",
            default=0.0,
            type=float,
            metavar="D",
            help="dropout probability after ReLU in FFN",
        )
        parser.add_argument(
            "--input-dropout",
            type=float,
            metavar="D",
            help="dropout probability of the inputs",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-output-dim",
            type=int,
            metavar="N",
            help="decoder output dimension",
        )
        parser.add_argument(
            "--decoder-input-dim", type=int, metavar="N", help="decoder input dimension"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads or LightConv/DynamicConv heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            default=False,
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--adaptive-softmax-cutoff",
            metavar="EXPR",
            help="comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion",
        )
        parser.add_argument(
            "--adaptive-softmax-dropout",
            type=float,
            metavar="D",
            help="sets adaptive softmax dropout for the tail projections",
        )
        parser.add_argument(
            "--adaptive-softmax-factor",
            type=float,
            metavar="N",
            help="adaptive input factor",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            default=False,
            action="store_true",
            help="if set, disables positional embeddings (outside self attention)",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            default=False,
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--character-embeddings",
            default=False,
            action="store_true",
            help="if set, uses character embedding convolutions to produce token embeddings",
        )
        parser.add_argument(
            "--character-filters",
            type=str,
            metavar="LIST",
            default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
            help="size of character embeddings",
        )
        parser.add_argument(
            "--character-embedding-dim",
            type=int,
            metavar="N",
            default=4,
            help="size of character embeddings",
        )
        parser.add_argument(
            "--char-embedder-highway-layers",
            type=int,
            metavar="N",
            default=2,
            help="number of highway layers for character token embeddder",
        )
        parser.add_argument(
            "--adaptive-input",
            default=False,
            action="store_true",
            help="if set, uses adaptive input",
        )
        parser.add_argument(
            "--adaptive-input-factor",
            type=float,
            metavar="N",
            help="adaptive input factor",
        )
        parser.add_argument(
            "--adaptive-input-cutoff",
            metavar="EXPR",
            help="comma separated list of adaptive input cutoff points.",
        )
        parser.add_argument(
            "--tie-adaptive-weights",
            action="store_true",
            help="if set, ties the weights of adaptive softmax and adaptive input",
        )
        parser.add_argument(
            "--tie-adaptive-proj",
            action="store_true",
            help="if set, ties the projection weights of adaptive softmax and adaptive input",
        )
        parser.add_argument(
            "--decoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the decoder",
        )

        """LightConv and DynamicConv arguments"""
        parser.add_argument(
            "--decoder-kernel-size-list",
            type=lambda x: utils.eval_str_list(x, int),
            help='list of kernel size (default: "[3,7,15,31,31,31]")',
        )
        parser.add_argument(
            "--decoder-glu", type=utils.eval_bool, help="glu after in proj"
        )
        parser.add_argument(
            "--decoder-conv-type",
            default="dynamic",
            type=str,
            choices=["dynamic", "lightweight"],
            help="type of convolution",
        )
        parser.add_argument("--weight-softmax", default=True, type=utils.eval_bool)
        parser.add_argument(
            "--weight-dropout",
            type=float,
            metavar="D",
            help="dropout probability for conv weights",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = args.tokens_per_sample
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = args.tokens_per_sample

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.dictionary),
                task.dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                utils.eval_str_list(args.adaptive_input_cutoff, type=int),
            )
        else:
            embed_tokens = Embedding(
                len(task.dictionary), args.decoder_input_dim, task.dictionary.pad()
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = LightConvDecoder(
            args,
            task.output_dictionary,
            embed_tokens,
            no_encoder_attn=True,
            final_norm=False,
        )
        return LightConvLanguageModel(decoder)


@register_model_architecture("lightconv_lm", "lightconv_lm")
def base_lm_architecture(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.decoder_conv_dim = getattr(args, "decoder_conv_dim", args.decoder_embed_dim)

    # The model training is not stable without this
    args.decoder_normalize_before = True

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.decoder_kernel_size_list = getattr(
        args, "decoder_kernel_size_list", [3, 7, 15, 31, 31, 31]
    )
    if len(args.decoder_kernel_size_list) == 1:
        args.decoder_kernel_size_list = (
            args.decoder_kernel_size_list * args.decoder_layers
        )
    assert (
        len(args.decoder_kernel_size_list) == args.decoder_layers
    ), "decoder_kernel_size_list doesn't match decoder_layers"
    args.decoder_glu = getattr(args, "decoder_glu", True)
    args.input_dropout = getattr(args, "input_dropout", 0.1)
    args.weight_dropout = getattr(args, "weight_dropout", args.attention_dropout)


@register_model_architecture("lightconv_lm", "lightconv_lm_gbw")
def lightconv_lm_gbw(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)
