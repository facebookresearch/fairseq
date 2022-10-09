# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import register_model, register_model_architecture
from fairseq.models.multilingual_transformer import MultilingualTransformerModel
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)
from fairseq.utils import safe_hasattr

from .latent_transformer import LatentTransformerDecoder, LatentTransformerEncoder


@register_model("latent_multilingual_transformer")
class LatentMultilingualTransformerModel(MultilingualTransformerModel):
    """A variant of standard multilingual Transformer models which encoder and/or
    decoders supports latent depth, as is in "Deep Transformer with Latent Depth"
    (https://arxiv.org/abs/2009.13102).
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        MultilingualTransformerModel.add_args(parser)
        parser.add_argument(
            '--soft-select',
            action='store_true',
            help='use soft samples in training an inference',
        )
        parser.add_argument(
            '--sampling-tau',
            type=float,
            default=5.,
            help='sampling temperature',
        )

    @classmethod
    def _get_module_class(cls, is_encoder, args, lang_dict, embed_tokens, langs):
        if is_encoder:
            if safe_hasattr(args, "encoder_latent_layer") and args.encoder_latent_layer:
                return LatentTransformerEncoder(
                    args, lang_dict, embed_tokens, num_logits=len(langs)
                )
            else:
                return TransformerEncoder(args, lang_dict, embed_tokens)
        else:
            if safe_hasattr(args, "decoder_latent_layer") and args.decoder_latent_layer:
                return LatentTransformerDecoder(
                    args, lang_dict, embed_tokens, num_logits=len(langs)
                )
            else:
                return TransformerDecoder(args, lang_dict, embed_tokens)


@register_model_architecture(
    "latent_multilingual_transformer", "latent_multilingual_transformer"
)
def latent_multilingual_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.share_encoders = getattr(args, "share_encoders", True)
    args.share_decoders = getattr(args, "share_decoders", True)
    args.share_encoder_embeddings = getattr(args, "share_encoder_embeddings", True)
    args.share_decoder_embeddings = getattr(args, "share_decoder_embeddings", True)

    base_architecture(args)
