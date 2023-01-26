# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils
from fairseq.models import (
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
    base_architecture,
)
from fairseq.models.multilingual_transformer import (
    MultilingualTransformerModel,
    base_multilingual_architecture,
)
from fairseq.utils import safe_hasattr
from collections import OrderedDict


@register_model("multilingual_transformer_from_mbart")
class MultilingualTransformerModelFromMbart(MultilingualTransformerModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if not safe_hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not safe_hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split("-")[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split("-")[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=src_langs,
                    embed_dim=args.encoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.encoder_embed_path,
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embed_dim=args.decoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.decoder_embed_path,
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.encoder_embed_dim,
                        args.encoder_embed_path,
                    )
                lang_encoders[lang] = MultilingualTransformerModel._get_module_class(
                    True, args, task.dicts[lang], encoder_embed_tokens, src_langs
                )
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.decoder_embed_dim,
                        args.decoder_embed_path,
                    )
                lang_decoders[lang] = MultilingualTransformerModel._get_module_class(
                    False, args, task.dicts[lang], decoder_embed_tokens, tgt_langs
                )
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = (
                shared_encoder if shared_encoder is not None else get_encoder(src)
            )
            decoders[lang_pair] = (
                shared_decoder if shared_decoder is not None else get_decoder(tgt)
            )

        return MultilingualTransformerModelFromMbart(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        state_dict_subset = state_dict.copy()
        lang_pairs = set([x.split(".")[1] for x in state_dict.keys()])
        finetune_mode = not any("neutral" in lp for lp in lang_pairs)

        if finetune_mode:
            # load a pre-trained mBART/BART model
            # we need this code because mBART/BART are not of type FairseqMultiModel but FairseqModel
            # so we hackishly load the weights by replicating them for all lang pairs
            print("loading pre-trained BART")
            self_state_dict = self.state_dict()
            for k, v in state_dict.items():
                for lang_pair in self.models:
                    new_key = k if "models." in k else f"models.{lang_pair}.{k}"
                    # print(new_key)
                    if self_state_dict[new_key].shape == v.shape:
                        state_dict_subset[new_key] = v
                    elif any(
                        w in k
                        for w in [
                            "encoder.embed_tokens.weight",
                            "decoder.embed_tokens.weight",
                            "decoder.output_projection.weight",
                        ]
                    ):
                        # why vocab_size - 5? because there are `vocab_size` tokens from the language
                        # and 5 additional tokens in the denoising task: eos,bos,pad,unk,mask.
                        # but in the translation task there are only `vocab_size` + 4 (no mask).
                        print(
                            f"{k}: {self_state_dict[new_key].shape} != {v.shape}",
                            end="",
                            flush=True,
                        )
                        vocab_size = v.shape[0] - 5
                        state_dict_subset[new_key] = self_state_dict[new_key]
                        state_dict_subset[new_key] = v[: vocab_size + 4]
                        print(f" => fixed by using first {vocab_size + 4} dims")
                    else:
                        raise ValueError("unable to load model due to mimatched dims!")
                del state_dict_subset[k]
        else:
            print("loading pre-trained emotion translation model")
            for k, _ in state_dict.items():
                assert k.startswith("models.")
                lang_pair = k.split(".")[1]
                if lang_pair not in self.models:
                    del state_dict_subset[k]

        super().load_state_dict(state_dict_subset, strict=strict, model_cfg=model_cfg)


@register_model_architecture("transformer", "transformer_small")
def transformer_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)


@register_model_architecture(
    "multilingual_transformer_from_mbart", "multilingual_small"
)
def multilingual_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_multilingual_architecture(args)
