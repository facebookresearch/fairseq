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
from fairseq.models.fconv import FConvDecoder
from fairseq.utils import safe_hasattr


@register_model("fconv_lm")
class FConvLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-layers",
            type=str,
            metavar="EXPR",
            help="decoder layers [(dim, kernel_size), ...]",
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="decoder output embedding dimension",
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
            "--decoder-attention",
            type=str,
            metavar="EXPR",
            help="decoder attention [True, ...]",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if safe_hasattr(args, "max_target_positions") and not safe_hasattr(
            args, "tokens_per_sample"
        ):
            args.tokens_per_sample = args.max_target_positions

        decoder = FConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.tokens_per_sample,
            share_embed=False,
            positional_embeddings=False,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            adaptive_softmax_dropout=args.adaptive_softmax_dropout,
        )
        return FConvLanguageModel(decoder)


@register_model_architecture("fconv_lm", "fconv_lm")
def base_lm_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_layers = getattr(args, "decoder_layers", "[(1268, 4)] * 13")
    args.decoder_attention = getattr(args, "decoder_attention", "False")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)


@register_model_architecture("fconv_lm", "fconv_lm_dauphin_wikitext103")
def fconv_lm_dauphin_wikitext103(args):
    layers = "[(850, 6)] * 3"
    layers += " + [(850, 1)] * 1"
    layers += " + [(850, 5)] * 4"
    layers += " + [(850, 1)] * 1"
    layers += " + [(850, 4)] * 3"
    layers += " + [(1024, 4)] * 1"
    layers += " + [(2048, 4)] * 1"
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 280)
    args.decoder_layers = getattr(args, "decoder_layers", layers)
    args.decoder_attention = getattr(args, "decoder_attention", "False")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,20000,200000"
    )
    base_lm_architecture(args)


@register_model_architecture("fconv_lm", "fconv_lm_dauphin_gbw")
def fconv_lm_dauphin_gbw(args):
    layers = "[(512, 5)]"
    layers += " + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3"
    layers += " + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3"
    layers += " + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6"
    layers += " + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]"
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_layers = getattr(args, "decoder_layers", layers)
    args.decoder_attention = getattr(args, "decoder_attention", "False")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )
    base_lm_architecture(args)
