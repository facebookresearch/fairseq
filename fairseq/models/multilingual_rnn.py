# Copyright (c) Facebook, Inc. and its affiliates.
# Developed by clefourrier
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from dataclasses import dataclass, field
from torch.nn import Embedding

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.rnn import RNNEncoder, RNNDecoder, RNNModelConfig, \
    DEFAULT_MAX_TARGET_POSITIONS, DEFAULT_MAX_SOURCE_POSITIONS


@dataclass
class MultilingualRNNModelConfig(RNNModelConfig):
    share_encoder_embeddings: bool = field(
        default=False,
        metadata={"help": "share encoder embeddings across all source languages"}
    )
    share_decoder_embeddings: bool = field(
        default=False,
        metadata={"help": "share decoder embeddings across all source languages"}
    )
    share_encoders: bool = field(
        default=False,
        metadata={"help": "share encoder embeddings across all source languages"}
    )
    share_decoders: bool = field(
        default=False,
        metadata={"help": "share encoder embeddings across all source languages"}
    )


@register_model('multilingual_rnn', dataclass=MultilingualRNNModelConfig)
class MultilingualRNNModel(FairseqMultiModel):
    """Train RNN models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from rnn.RNNModel and assume that all language
    pairs use a single RNN architecture, but the model weights will be shared
    across language pairs if they have the same model architecture.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_model(cls, cfg: MultilingualRNNModelConfig, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        assert isinstance(task, MultilingualTranslationTask)

        if not hasattr(cfg, 'max_source_positions'):
            cfg.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(cfg, 'max_target_positions'):
            cfg.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

        if cfg.share_encoders:
            cfg.share_encoder_embeddings = True
        if cfg.share_decoders:
            cfg.share_decoder_embeddings = True

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
        if cfg.share_all_embeddings:
            if cfg.encoder_embed_dim != cfg.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if cfg.decoder_embed_path and (
                    cfg.decoder_embed_path != cfg.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=cfg.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=cfg.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            if cfg.share_encoder_embeddings:
                shared_encoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=src_langs,
                        embed_dim=cfg.encoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=cfg.encoder_embed_path,
                    )
                )
            if cfg.share_decoder_embeddings:
                shared_decoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=tgt_langs,
                        embed_dim=cfg.decoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=cfg.decoder_embed_path,
                    )
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang], cfg.encoder_embed_dim, cfg.encoder_embed_path
                    )
                lang_encoders[lang] = RNNEncoder(
                    dictionary=task.dicts[lang],
                    embed_dim=cfg.encoder_embed_dim,
                    hidden_size=cfg.encoder_hidden_size,
                    num_layers=cfg.encoder_layers,
                    dropout_in=(cfg.encoder_dropout_in if cfg.encoder_dropout_in >= 0 else cfg.dropout),
                    dropout_out=(cfg.encoder_dropout_out if cfg.encoder_dropout_out >= 0 else cfg.dropout),
                    bidirectional=cfg.encoder_bidirectional,
                    pretrained_embed=encoder_embed_tokens,
                    rnn_type=cfg.rnn_type,
                    max_source_positions=cfg.max_source_positions)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], cfg.decoder_embed_dim, cfg.decoder_embed_path
                    )

                lang_decoders[lang] = RNNDecoder(
                    dictionary=task.dicts[lang],
                    embed_dim=cfg.decoder_embed_dim,
                    hidden_size=cfg.decoder_hidden_size,
                    out_embed_dim=cfg.decoder_out_embed_dim,
                    num_layers=cfg.decoder_layers,
                    attention_type=cfg.attention_type,
                    dropout_in=(cfg.decoder_dropout_in if cfg.decoder_dropout_in >= 0 else cfg.dropout),
                    dropout_out=(cfg.decoder_dropout_out if cfg.decoder_dropout_out >= 0 else cfg.dropout),
                    rnn_type=cfg.rnn_type,
                    encoder_output_units=cfg.encoder_hidden_size,
                    pretrained_embed=decoder_embed_tokens,
                    share_input_output_embed=cfg.share_decoder_input_output_embed,
                    adaptive_softmax_cutoff=(
                        utils.eval_str_list(cfg.adaptive_softmax_cutoff,
                                            type=int)
                        if cfg.criterion == "adaptive_loss"
                        else None
                    ),
                    max_target_positions=cfg.max_target_positions,
                    residuals=False,
                )
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if cfg.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if cfg.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src)
            decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt)

        return MultilingualRNNModel(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True, args="none"):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith("models.")
            lang_pair = k.split(".")[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict, args=args)


@register_model_architecture('multilingual_rnn', 'multilingual_lstm')
def multilingual_lstm(cfg):
    cfg.encoder_bidirectional = False
    cfg.rnn_type = "lstm"


@register_model_architecture('multilingual_rnn', 'multilingual_gru')
def multilingual_gru(cfg):
    cfg.encoder_bidirectional = False
    cfg.rnn_type = "gru"


@register_model_architecture('multilingual_rnn', 'multilingual_bilstm')
def multilingual_bilstm(cfg):
    cfg.encoder_bidirectional = True
    cfg.rnn_type = "lstm"


@register_model_architecture('multilingual_rnn', 'multilingual_bigru')
def multilingual_bigru(cfg):
    cfg.encoder_bidirectional = True
    cfg.rnn_type = "gru"
