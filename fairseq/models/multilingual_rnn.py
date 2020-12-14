# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from torch.nn import Embedding

from fairseq import utils
from fairseq.models import (
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from fairseq.models import lstm
from fairseq.models.rnn import RNNEncoder, RNNDecoder, RNNModel, base_architecture, DEFAULT_MAX_TARGET_POSITIONS, DEFAULT_MAX_SOURCE_POSITIONS


@register_model('multilingual_rnn')
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

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        RNNModel.add_args(parser)
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_multilingual_rnn(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

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
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
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
                shared_encoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=src_langs,
                        embed_dim=args.encoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.encoder_embed_path,
                    )
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=tgt_langs,
                        embed_dim=args.decoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.decoder_embed_path,
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
                        task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path
                    )
                lang_encoders[lang] = RNNEncoder(
                    dictionary=task.dicts[lang],
                    embed_dim=args.encoder_embed_dim,
                    hidden_size=args.encoder_hidden_size,
                    num_layers=args.encoder_layers,
                    dropout_in=args.encoder_dropout_in,
                    dropout_out=args.encoder_dropout_out,
                    bidirectional=args.encoder_bidirectional,
                    pretrained_embed=encoder_embed_tokens,
                    rnn_type=args.rnn_type,
                    max_source_positions=args.max_source_positions)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim, args.decoder_embed_path
                    )

                lang_decoders[lang] = RNNDecoder(
                    dictionary=task.dicts[lang],
                    embed_dim=args.decoder_embed_dim,
                    hidden_size=args.decoder_hidden_size,
                    out_embed_dim=args.decoder_out_embed_dim,
                    num_layers=args.decoder_layers,
                    dropout_in=args.decoder_dropout_in,
                    dropout_out=args.decoder_dropout_out,
                    attention=utils.eval_bool(args.decoder_attention),
                    attention_type=args.attention_type,
                    rnn_type=args.rnn_type,
                    encoder_output_units=args.encoder_hidden_size,
                    pretrained_embed=decoder_embed_tokens,
                    share_input_output_embed=args.share_decoder_input_output_embed,
                    adaptive_softmax_cutoff=(
                        utils.eval_str_list(args.adaptive_softmax_cutoff,
                                            type=int)
                        if args.criterion == "adaptive_loss"
                        else None
                    ),
                    max_target_positions=args.max_target_positions,
                    residuals=False,
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
            encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src)
            decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt)

        return MultilingualRNNModel(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None): #, args=None
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith("models.")
            lang_pair = k.split(".")[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict, model_cfg=model_cfg)


@register_model_architecture('multilingual_rnn', 'multilingual_rnn_base')
def base_multilingual_rnn(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', False)


@register_model_architecture('multilingual_rnn', 'multilingual_lstm')
def multilingual_lstm(args):
    args.encoder_bidirectional = False
    args.rnn_type = "lstm"
    base_multilingual_rnn(args)

@register_model_architecture('multilingual_rnn', 'multilingual_gru')
def multilingual_lstm(args):
    args.encoder_bidirectional = False
    args.rnn_type = "gru"
    base_multilingual_rnn(args)

@register_model_architecture('multilingual_rnn', 'multilingual_bilstm')
def multilingual_lstm(args):
    args.encoder_bidirectional = True
    args.rnn_type = "lstm"
    base_multilingual_rnn(args)

@register_model_architecture('multilingual_rnn', 'multilingual_bigru')
def multilingual_lstm(args):
    args.encoder_bidirectional = True
    args.rnn_type = "gru"
    base_multilingual_rnn(args)