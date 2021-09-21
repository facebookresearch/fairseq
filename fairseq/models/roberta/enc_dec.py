import argparse
import logging

import torch.nn as nn
import fairseq.checkpoint_utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.roberta import model as roberta

logger = logging.getLogger(__name__)


@register_model("roberta_enc_dec")
class RobertaEncDecModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--pretrained-mlm-checkpoint",
            default=None,
            type=str,
            metavar="PRETRAINED",
            help="path to pretrained mlm checkpoint",
        )
        parser.add_argument(
            "--pretrained-decoder", action="store_true", help="reload decoder"
        )
        parser.add_argument(
            "--hack-layernorm-embedding",
            action="store_true",
            help="hack to reload old models trained with encoder-normalize-before=False (no equivalent to encoder-normalize-before=False and layernorm_embedding=False",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--share-all-embeddings",
            action="store_true",
            help="share encoder, decoder and output embeddings"
            " (requires shared dictionary and embed dim)",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_enc_dec_architecture(args)
        if args.pretrained_mlm_checkpoint:
            arg_overrides = None
            if args.hack_layernorm_embedding:
                arg_overrides = {"layernorm_embedding": False}
            loaded = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [args.pretrained_mlm_checkpoint], arg_overrides=arg_overrides
            )
            ([roberta_enc], _cfg, _task) = loaded
        else:
            # Do we need to edit untie_weights here ?
            share_in_out = (
                args.share_decoder_input_output_embed or args.share_all_embeddings
            )
            args.untie_weights_roberta = not share_in_out
            if args.hack_layernorm_embedding:
                args.layernorm_embedding = False
                args.encoder_normalize_before = False
            roberta_enc = roberta.RobertaModel.build_model(args, task)

        return cls.from_roberta(roberta_enc, args, task.source_dictionary)

    @staticmethod
    def from_roberta(roberta_enc: roberta.RobertaModel, args, dictionary):
        encoder = roberta_enc.encoder.sentence_encoder
        vocab_size, embed_dim = encoder.embed_tokens.weight.shape

        if args.share_all_embeddings:
            lm_head = roberta_enc.encoder.lm_head
            assert encoder.embed_tokens.weight is lm_head.weight, (
                "Can't use --share-all-embeddings with a model "
                "that was pretraiend with --untie-weights-roberta_enc"
            )
        else:
            lm_head = roberta.RobertaLMHead(
                embed_dim, vocab_size, roberta_enc.args.activation_fn
            )

        dec_embs = nn.Embedding(vocab_size, embed_dim, dictionary.pad())
        if args.share_all_embeddings or args.share_decoder_input_output_embed:
            # Note: I wasn't able to use Embedding _weight parameter to achive this sharing.
            dec_embs.weight = lm_head.weight

        decoder = TransformerDecoder(
            RobertaEncDecModel.read_args_from_roberta(roberta_enc.args),
            dictionary,
            dec_embs,
            no_encoder_attn=False,
            output_projection=lm_head,
        )
        if getattr(args, "pretrained_decoder", False):
            decoder_dict = encoder.state_dict()

            # TODO: hide setting "encoder_attn" layers behind a flag.
            for k, w in list(decoder_dict.items()):
                if ".self_attn" in k:
                    k_enc_attn = k.replace(".self_attn", ".encoder_attn")
                    decoder_dict[k_enc_attn] = w.detach().clone()

            for k, w in lm_head.state_dict().items():
                decoder_dict["output_projection." + k] = w

            missing_keys, unexpected_keys = decoder.load_state_dict(
                decoder_dict, strict=False
            )
            # missing_keys = [m for m in missing_keys if ".encoder_attn" not in m]
            assert not missing_keys and not unexpected_keys, (
                "Failed to load state dict. "
                f"Missing keys: {missing_keys}. "
                f"Unexpected keys: {unexpected_keys}."
            )

        if args.share_all_embeddings:
            assert decoder.output_projection.weight is decoder.embed_tokens.weight
            assert encoder.embed_tokens.weight is decoder.embed_tokens.weight
        elif args.share_decoder_input_output_embed:
            assert decoder.output_projection.weight is decoder.embed_tokens.weight
            assert encoder.embed_tokens.weight is not decoder.embed_tokens.weight
        else:
            assert decoder.output_projection.weight is not decoder.embed_tokens.weight
            assert encoder.embed_tokens.weight is not decoder.embed_tokens.weight

        return RobertaEncDecModel(encoder, decoder)

    @staticmethod
    def read_args_from_roberta(roberta_args: argparse.Namespace):
        # TODO: this would become easier if encoder/decoder where using a similar
        # TransformerConfig object
        args = argparse.Namespace(**vars(roberta_args))
        attr_map = [
            ("encoder_attention_heads", "decoder_attention_heads"),
            ("encoder_embed_dim", "decoder_embed_dim"),
            ("encoder_embed_dim", "decoder_output_dim"),
            ("encoder_normalize_before", "decoder_normalize_before"),
            ("encoder_layers_to_keep", "decoder_layers_to_keep"),
            ("encoder_ffn_embed_dim", "decoder_ffn_embed_dim"),
            ("encoder_layerdrop", "decoder_layerdrop"),
            ("encoder_layers", "decoder_layers"),
            ("encoder_learned_pos", "decoder_learned_pos"),
            # should this be set from here ?
            ("max_positions", "max_target_positions"),
        ]
        for k1, k2 in attr_map:
            setattr(args, k2, getattr(roberta_args, k1))

        args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
        args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
        args.share_decoder_input_output_embed = not roberta_args.untie_weights_roberta
        return args

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        super().upgrade_state_dict_named(state_dict, name)
        old_keys = list(state_dict.keys())

        # rename decoder -> encoder before upgrading children modules
        for k in old_keys:
            if k.startswith(prefix + "encoder.lm_head"):
                state_dict.pop(k)
                continue
            new_k = k
            new_k = new_k.replace(".sentence_encoder.", ".")
            new_k = new_k.replace("decoder.lm_head.", "decoder.output_projection.")
            if k == new_k:
                continue
            # print(k, "->", new_k)
            state_dict[new_k] = state_dict.pop(k)


@register_model_architecture("roberta_enc_dec", "roberta_enc_dec")
def base_enc_dec_architecture(args):
    args.hack_layernorm_embedding = getattr(args, "hack_layernorm_embedding", False)
    args.pretrained_mlm_checkpoint = getattr(args, "pretrained_mlm_checkpoint", None)
    args.pretrained_decoder = getattr(args, "pretrained_decoder", None)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )

    roberta.base_architecture(args)
