# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import OrderedDict, namedtuple

import torch.nn as nn

from fairseq import checkpoint_utils, utils
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.file_io import PathManager
from fairseq.models import register_model, register_model_architecture
from fairseq.models.speech_to_text import (
    SpeechWavTransformerEncoder,
    StackedSpeechWavTransformerEncoder,
    TransformerDecoder,
)
from fairseq.models.transformer import TransformerEncoder

from .s2t_dualinputtransformer import (
    DualInputEncoder,
    DualInputS2TTransformerModel,
    TransformerMultiInputDecoder,
)

logger = logging.getLogger(__name__)


@register_model("dual_input_wav_transformer")
class DualInputWavTransformerModel(DualInputS2TTransformerModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        def add_transformer_args(parser):
            # We can't use TransformerModel.add_args(parser), since it defines max-source-positions which is duplicated with tasks/speech_to_text.py
            # Transformer
            parser.add_argument(
                "--activation-fn",
                type=str,
                default="relu",
                choices=utils.get_available_activation_fns(),
                help="activation function to use",
            )
            parser.add_argument(
                "--dropout", type=float, metavar="D", help="dropout probability"
            )
            parser.add_argument(
                "--attention-dropout",
                type=float,
                metavar="D",
                help="dropout probability for attention weights",
            )
            parser.add_argument(
                "--activation-dropout",
                "--relu-dropout",
                type=float,
                metavar="D",
                help="dropout probability after activation in FFN.",
            )
            parser.add_argument(
                "--encoder-embed-dim",
                type=int,
                metavar="N",
                help="encoder embedding dimension",
            )
            parser.add_argument(
                "--encoder-ffn-embed-dim",
                type=int,
                metavar="N",
                help="encoder embedding dimension for FFN",
            )
            parser.add_argument(
                "--encoder-layers", type=int, metavar="N", help="num encoder layers"
            )
            parser.add_argument(
                "--encoder-attention-heads",
                type=int,
                metavar="N",
                help="num encoder attention heads",
            )
            parser.add_argument(
                "--encoder-normalize-before",
                action="store_true",
                help="apply layernorm before each encoder block",
            )
            parser.add_argument(
                "--decoder-embed-dim",
                type=int,
                metavar="N",
                help="decoder embedding dimension",
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
                help="num decoder attention heads",
            )
            parser.add_argument(
                "--decoder-normalize-before",
                action="store_true",
                help="apply layernorm before each decoder block",
            )
            parser.add_argument(
                "--share-decoder-input-output-embed",
                action="store_true",
                help="share decoder input and output embeddings",
            )
            parser.add_argument(
                "--layernorm-embedding",
                action="store_true",
                help="add layernorm to embedding",
            )
            parser.add_argument(
                "--no-scale-embedding",
                action="store_true",
                help="if True, dont scale embeddings",
            )

            parser.add_argument(
                "--encoder-learned-pos",
                action="store_true",
                help="use learned positional embeddings",
            )
            parser.add_argument(
                "--decoder-learned-pos",
                action="store_true",
                help="use learned positional embeddings",
            )

        add_transformer_args(parser)
        SpeechWavTransformerEncoder.add_args(parser)
        parser.add_argument(
            "--load-pretrained-speech-text-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained speech text encoder from SpeechTextPreTrainModel """,
        )
        parser.add_argument(
            "--load-pretrained-wav2vec-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained speech text encoder from wav2vec """,
        )

        parser.add_argument(
            "--load-pretrained-speech-text-decoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained speech text decoder from SpeechTextPreTrainModel """,
        )
        parser.add_argument(
            "--load-pretrained-text-decoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained  text decoder """,
        )
        parser.add_argument(
            "--load-init-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to load seed encoder model """,
        )
        parser.add_argument(
            "--load-init-decoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to load seed decoder model """,
        )

        parser.add_argument(
            "--text-input-cost-ratio",
            type=float,
            default=1.0,
            metavar="V",
            help="text input cost ratio relative to speech input cost",
        )
        parser.add_argument(
            "--enc-grad-mult",
            type=float,
            metavar="V",
            default=1.0,
            help="multiply enc1 and enc2 gradient by V",
        )
        parser.add_argument(
            "--enc2-along-grad-mult",
            type=float,
            metavar="V",
            default=1.0,
            help="multiply enc2 gradient by V if only enc2 is used",
        )
        parser.add_argument(
            "--no-strict-check-pretrain-model",
            action="store_true",
            help="Don't apply strict model check for the pretrained model",
        )

        parser.add_argument(
            "--stacked-encoder",
            action="store_true",
            help="stack speech and text encoders",
        )

    @classmethod
    def update_transformer_encoder_cfg(cls, args, update_dict):
        cfg = dict(args._get_kwargs())
        for fkey in update_dict.keys():
            cfg[fkey] = update_dict[fkey]
        cfg.pop("_name", None)  # remove keys start with _
        model_args = namedtuple("args", cfg.keys())(*cfg.values())
        return model_args

    @classmethod
    def build_text_encoder(cls, args, src_dictionary):
        enc_emb = nn.Embedding(
            len(src_dictionary), args.encoder_embed_dim, src_dictionary.pad()
        )
        model_args = cls.update_transformer_encoder_cfg(
            args,
            {
                "encoder_layers": args.text_encoder_layers,
                "max_source_positions": args.max_positions_text,
            },
        )
        text_encoder = TransformerEncoder(model_args, src_dictionary, enc_emb)
        return text_encoder

    @classmethod
    def build_speech_encoder(cls, args):
        model_args = cls.update_transformer_encoder_cfg(
            args, {"encoder_layers": args.speech_encoder_layers}
        )
        speech_encoder = SpeechWavTransformerEncoder(model_args)
        return speech_encoder

    @classmethod
    def check_args(cls, condition, is_strict, msg):
        if condition:
            return
        if is_strict:
            raise ValueError(msg)
        logger.warn(msg)

    @classmethod
    def build_encoder(cls, args, task):
        # text_encoder = cls.build_text_encoder(args, task.source_dictionary )
        text_encoder = cls.build_text_encoder(args, task.src_dict)
        speech_encoder = cls.build_speech_encoder(args)
        if args.load_pretrained_wav2vec_encoder:
            component_pairs = (
                ("feature_extractor", speech_encoder.subsample),
                ("post_extract_proj", speech_encoder.feat_proj),
                ("layer_norm", speech_encoder.feat_layer_norm),
                ("encoder.pos_conv", speech_encoder.embed_positions),
                ("encoder.layers", speech_encoder.layers),
                ("encoder.layer_norm", speech_encoder.layer_norm),
                ("mask_emb", speech_encoder.mask_emb),
            )
            state = cls.load_pretrained_speech_text_components(
                args.load_pretrained_wav2vec_encoder, component_pairs
            )
            cls.check_args(
                args.encoder_normalize_before
                == state["cfg"]["model"]["layer_norm_first"],
                not args.no_strict_check_pretrain_model,
                f"encoder_normalize_before {args.encoder_normalize_before} doesn't match with the pretrained model",
            )
            cls.check_args(
                args.activation_fn == state["cfg"]["model"]["activation_fn"],
                not args.no_strict_check_pretrain_model,
                f"activation_fn {args.activation_fn} doesn't match with the pretrained model",
            )

        if getattr(args, "stacked_encoder", False):
            if args.encoder_shared_text_layers_from_begin > 0:
                raise ValueError(
                    "We can not stack encoders and share encoders at the same time!"
                )
            speech_encoder = StackedSpeechWavTransformerEncoder(
                speech_encoder, text_encoder.layers, text_encoder.layer_norm
            )
        else:
            cls.share_speech_text_encoder(
                speech_encoder, text_encoder, args.encoder_shared_text_layers_from_begin
            )

        cross_attentive_loss_before_last_layer = (
            0 if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else -1
        )
        encoder = DualInputEncoder(
            args,
            speech_encoder,
            text_encoder,
            task.src_dict,
            cross_attentive_loss_before_last_layer,
        )
        if args.load_pretrained_speech_text_encoder:
            component_pairs = (
                ("encoder.sup_s2s_speech_encoder", encoder.spch_encoder),
                ("encoder.text_encoder", encoder.text_encoder),
            )
            cls.load_pretrained_speech_text_components(
                args.load_pretrained_speech_text_encoder, component_pairs
            )
        if getattr(args, "load_init_encoder", "") != "":
            checkpoint_utils.load_pretrained_component_from_model(
                encoder, args.load_init_encoder
            )
        return encoder

    @classmethod
    def build_text_decoder(cls, args, tgt_dictionary, dec_emb_share=None):
        dec_emb = (
            nn.Embedding(
                len(tgt_dictionary), args.decoder_embed_dim, tgt_dictionary.pad()
            )
            if dec_emb_share is None
            else dec_emb_share
        )
        text_decoder = TransformerDecoder(args, tgt_dictionary, dec_emb)
        return text_decoder

    @classmethod
    def build_decoder(cls, args, task):
        text_decoder = cls.build_text_decoder(args, task.target_dictionary)
        compute_cross_attentive_loss = (
            True if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else False
        )
        cross_attentive_loss_without_norm = getattr(
            args, "attentive_cost_without_normalize", False
        )
        cross_attentive_loss_reverse = (
            False  # getattr(args, "attentive_cost_reverse", False)
        )
        if getattr(args, "load_pretrained_text_decoder", "") != "":
            checkpoint_utils.load_pretrained_component_from_model(
                text_decoder, args.load_pretrained_text_decoder
            )

        if args.load_pretrained_speech_text_decoder:
            component_pairs = (("decoder.text_decoder", text_decoder),)
            cls.load_pretrained_speech_text_components(
                args.load_pretrained_speech_text_decoder, component_pairs
            )

        decoder = TransformerMultiInputDecoder(
            dictionary=task.target_dictionary,
            spch_decoder=text_decoder,
            text_decoder=text_decoder,
            compute_cross_attentive_loss=compute_cross_attentive_loss,
            cross_attentive_loss_with_norm=True
            if not cross_attentive_loss_without_norm
            else False,
            cross_attentive_loss_reverse=cross_attentive_loss_reverse,
        )
        if getattr(args, "load_init_decoder", "") != "":
            checkpoint_utils.load_pretrained_component_from_model(
                decoder, args.load_init_decoder
            )
        return decoder

    @classmethod
    def load_pretrained_speech_text_components(cls, checkpoint, component_pairs):
        if not PathManager.exists(checkpoint):
            raise IOError("Model file not found: {}".format(checkpoint))
        state = load_checkpoint_to_cpu(checkpoint)
        for component_type, component in component_pairs:
            if isinstance(component, nn.parameter.Parameter):
                component.data.copy_(state["model"][component_type])
            else:
                component_state_dict = OrderedDict()
                for key in state["model"].keys():
                    if key.startswith(component_type):
                        component_subkey = key[len(component_type) + 1 :]
                        component_state_dict[component_subkey] = state["model"][key]
                component.load_state_dict(component_state_dict, strict=True)
        return state

    @classmethod
    def share_speech_text_encoder(
        cls, speech_encoder, text_encoder, shared_layers_from_begin
    ):
        if shared_layers_from_begin > 0:
            num_text_encoder_layers = len(text_encoder.layers)
            assert len(speech_encoder.layers) >= shared_layers_from_begin
            assert num_text_encoder_layers >= shared_layers_from_begin
            assert len(speech_encoder.layers) >= num_text_encoder_layers
            for i, ly in enumerate(
                speech_encoder.layers[
                    -num_text_encoder_layers : -num_text_encoder_layers
                    + shared_layers_from_begin
                ]
            ):
                assert isinstance(text_encoder.layers[i], type(ly))
                text_encoder.layers[i] = ly


@register_model_architecture(
    "dual_input_wav_transformer", "dualinputs2twavtransformer_base"
)
def dualinputs2twavtransformer_base(args):
    # speech masking
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)
    args.speech_mask_length = getattr(args, "speech_mask_length", 10)
    args.speech_mask_prob = getattr(args, "speech_mask_prob", 0.65)
    args.speech_mask_selection = getattr(args, "speech_mask_selection", "static")
    args.speech_mask_other = getattr(args, "speech_mask_other", 0)
    args.speech_mask_min_space = getattr(args, "speech_mask_min_space", 1)
    args.speech_no_mask_overlap = getattr(args, "speech_no_mask_overlap", False)
    args.speech_conv_bias = getattr(args, "speech_conv_bias", False)
    args.speech_extractor_mode = getattr(args, "speech_extractor_mode", "default")
    args.no_strict_check_pretrain_model = getattr(
        args, "no_strict_check_pretrain_model", False
    )

    args.speech_mask_channel_length = getattr(args, "speech_mask_channel_length", 10)
    args.speech_mask_channel_prob = getattr(args, "speech_mask_channel_prob", 0.0)
    args.speech_mask_channel_selection = getattr(
        args, "speech_mask_channel_selection", "static"
    )
    args.speech_mask_channel_other = getattr(args, "speech_mask_channel_other", 0)
    args.speech_mask_channel_min_space = getattr(
        args, "speech_mask_channel_min_space", 1
    )
    args.speech_no_mask_channel_overlap = getattr(
        args, "speech_no_mask_channel_overlap", False
    )
    args.no_scale_feature = getattr(args, "", False)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0.0)  # 0.1

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", args.encoder_embed_dim * 4
    )
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.1)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_attention_heads = getattr(
        args, "decoder_attention_heads", args.encoder_attention_heads
    )
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")  # gelu?
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
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
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 6
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)


@register_model_architecture(
    "dual_input_wav_transformer", "dualinputs2twavtransformer_base_stack"
)
def dualinputs2twavtransformer_base_stack(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 0
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.stacked_encoder = getattr(args, "stacked_encoder", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    dualinputs2twavtransformer_base(args)


@register_model_architecture(
    "dual_input_wav_transformer", "dualinputs2twavtransformer_large"
)
def dualinputs2twavtransformer_large(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 24)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 12)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 12
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    dualinputs2twavtransformer_base(args)
