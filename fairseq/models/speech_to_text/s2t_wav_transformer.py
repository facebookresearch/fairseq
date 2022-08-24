#!/usr/bin/env python3

from collections import OrderedDict
import math
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.file_io import PathManager
from fairseq.models.speech_to_text.s2t_transformer import S2TTransformerModel

import torch
from torch import nn

from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from fairseq.models.wav2vec import ConvFeatureExtractionModel
from fairseq.modules import GradMultiply, LayerNorm, SamePad, TransformerEncoderLayer


#   Transformer encoder with wave input, it is adopted from wav2vec 2.0 Encoder.
#       use wav input
#       use trained position embedding so it is easier to match with text input
class SpeechWavTransformerEncoder(FairseqEncoder):

    # extra parameters for speech encoder besides those defined in transformermodel
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--load-pretrained-wav2vec-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained speech text encoder from wav2vec """,
        )
        parser.add_argument(
            "--dropout-input",
            type=float,
            metavar="D",
            help="dropout to apply to the input (after feat extr)",
        )
        parser.add_argument(
            "--dropout-features",
            type=float,
            metavar="D",
            help="dropout to apply to the unmasked features (after feat extr)",
        )
        parser.add_argument(
            "--speech-extractor-mode",
            type=str,
            default="layer_norm",
            choices=["default", "layer_norm"],
            help="feature extractor norm",
        )

        parser.add_argument(
            "--speech-conv-bias",
            action="store_true",
            help="include bias in speech conv encoder",
        )

        parser.add_argument(
            "--conv-feature-layers",
            default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
            help="string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]",
        )

        parser.add_argument(
            "--speech-mask-length",
            type=int,
            help="repeat the mask indices multiple times",
        )

        parser.add_argument(
            "--speech-mask-prob",
            type=float,
            help="probability of replacing a token with mask",
        )

        parser.add_argument(
            "--speech-mask-selection",
            type=str,
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose masks",
        )

        parser.add_argument(
            "--speech-mask-other",
            type=float,
            help="stdev of the mask length in case of 'normal' selection strategy",
        )

        parser.add_argument(
            "--speech-no-mask-overlap",
            action="store_true",
            help="whether to allow masks to overlap",
        )

        parser.add_argument(
            "--speech-mask-min-space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
        )

        parser.add_argument(
            "--speech-mask-channel-length",
            type=int,
            help="repeat the mask indices multiple times",
        )

        parser.add_argument(
            "--speech-mask-channel-prob",
            type=float,
            help="probability of replacing a token with mask",
        )

        parser.add_argument(
            "--speech-mask-channel-selection",
            type=str,
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose masks",
        )

        parser.add_argument(
            "--speech-mask-channel-other",
            type=float,
            help="stdev of the mask length in case of 'normal' selection strategy",
        )

        parser.add_argument(
            "--speech-no-mask-channel-overlap",
            action="store_true",
            help="whether to allow masks to overlap",
        )

        parser.add_argument(
            "--no-scale-feature",
            action="store_true",
            help="no scale for the calculated features",
        )

        parser.add_argument(
            "--speech-mask-channel-min-space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
        )

        parser.add_argument(
            "--feature-grad-mult",
            type=float,
            help="reset feature grad mult in wav2vec 2.0 to this",
        )

        # positional embeddings
        parser.add_argument(
            "--conv-pos",
            type=int,
            default=128,
            help="number of filters for convolutional positional embeddings",
        )

        parser.add_argument(
            "--conv-pos-groups",
            type=int,
            default=16,
            help="number of groups for convolutional positional embedding",
        )
        # model configures
        parser.add_argument(
            "--speech-encoder-layers",
            type=int,
            help="number of speech encoder layers",
        )
        parser.add_argument(
            "--text-encoder-layers",
            type=int,
            help="number of text encoder layers",
        )

    def __init__(self, args, alway_mask=False):
        super().__init__(args)
        self.args = args
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.feat_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_feature:
            self.feat_scale = 1.0

        subsample = ConvFeatureExtractionModel(
            conv_layers=eval(args.conv_feature_layers),
            dropout=0.0,
            mode=args.speech_extractor_mode,  # default, layer_norm
            conv_bias=args.speech_conv_bias,
        )
        self.feature_enc_layers = eval(args.conv_feature_layers)
        self.subsample = subsample
        self.feat_proj = (
            nn.Linear(self.feature_enc_layers[-1][0], self.embedding_dim)
            if self.feature_enc_layers[-1][0] != self.embedding_dim
            else None
        )

        self.feat_layer_norm = LayerNorm(self.feature_enc_layers[-1][0])

        self.embed_positions = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        std = math.sqrt(4 / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.embed_positions.weight, mean=0, std=std)
        nn.init.constant_(self.embed_positions.bias, 0)

        self.embed_positions = nn.utils.weight_norm(
            self.embed_positions, name="weight", dim=2
        )
        self.embed_positions = nn.Sequential(
            self.embed_positions, SamePad(args.conv_pos), nn.GELU()
        )

        self.mask_prob = args.speech_mask_prob
        self.mask_selection = args.speech_mask_selection
        self.mask_other = args.speech_mask_other
        self.mask_length = args.speech_mask_length
        self.no_mask_overlap = args.speech_no_mask_overlap
        self.mask_min_space = args.speech_mask_min_space

        self.mask_channel_prob = args.speech_mask_channel_prob
        self.mask_channel_selection = args.speech_mask_channel_selection
        self.mask_channel_other = args.speech_mask_channel_other
        self.mask_channel_length = args.speech_mask_channel_length
        self.no_mask_channel_overlap = args.speech_no_mask_channel_overlap
        self.mask_channel_min_space = args.speech_mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)

        self.feature_grad_mult = args.feature_grad_mult

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(args.encoder_embed_dim).uniform_()
        )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_norm = LayerNorm(args.encoder_embed_dim)
        self.normalize_before = args.encoder_normalize_before
        self.alway_mask = alway_mask

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        for i in range(len(self.feature_enc_layers)):
            input_lengths = _conv_out_length(
                input_lengths,
                self.feature_enc_layers[i][1],
                self.feature_enc_layers[i][2],
            )

        return input_lengths.to(torch.long)

    def laod_pretrained_wav2vec(self, checkpoint):
        component_pairs = (
            ("feature_extractor", self.subsample),
            ("post_extract_proj", self.feat_proj),
            ("layer_norm", self.feat_layer_norm),
            ("encoder.pos_conv", self.embed_positions),
            ("encoder.layers", self.layers),
            ("encoder.layer_norm", self.layer_norm),
            ("mask_emb", self.mask_emb),
        )
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

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens=False,
        padding_mask=None,
        features_only=True,
    ):
        mask = self.training or self.alway_mask
        if self.feature_grad_mult > 0 and self.training:
            features = self.subsample(src_tokens)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.subsample(src_tokens)
        features = features.transpose(1, 2)
        features = self.feat_layer_norm(features)
        if self.feat_proj is not None:
            features = self.feat_proj(features)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
        else:
            input_lengths = src_lengths
        if input_lengths is not None:
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()

        features = self.feat_scale * features if self.feat_scale != 1.0 else features
        unmasked_features = features.clone()

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None

        def cal_transformer_layers(x, encoder_padding_mask, return_all_hiddens=False):
            # x: B x T x C
            positions = self.embed_positions(x.transpose(1, 2)).transpose(1, 2)
            x = x + positions
            if not self.normalize_before:
                x = self.layer_norm(x)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            encoder_states = []
            for layer in self.layers:
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)
            if self.normalize_before:
                x = self.layer_norm(x)
            return x, encoder_states

        x, encoder_states = cal_transformer_layers(x, padding_mask, return_all_hiddens)
        if features_only:
            return {
                "encoder_out": [x],  # [T x B x C]
                "encoder_padding_mask": [padding_mask]
                if padding_mask is not None
                else [],  # B x T
                "encoder_embedding": [],  #
                "encoder_states": encoder_states,  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [],
                "mask_indices": [mask_indices],
            }

        x_unmasked = x
        if self.mask_prob > 0 or self.mask_channel_prob > 0:
            x_unmasked, _ = cal_transformer_layers(unmasked_features, padding_mask)
        return {
            "encoder_out": [x],  # [T x B x C]
            "encoder_unmasked_out": [x_unmasked],  # [T x B x C]
            "encoder_padding_mask": [padding_mask]
            if padding_mask is not None
            else [],  # B x T
            "encoder_embedding": [],  #
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "mask_indices": [mask_indices] if mask_indices is not None else [],  # B X T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


class StackedSpeechWavTransformerEncoder(FairseqEncoder):
    def __init__(self, speech_enc, text_enc_layers, text_layer_norm):
        super().__init__(None)
        self.speech_encoder = speech_enc
        self.text_encoder_layers = text_enc_layers
        self.final_layer_norm = text_layer_norm

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        return_all_hiddens=False,
        padding_mask=None,
        features_only=True,
    ):

        out = self.speech_encoder.forward(
            src_tokens,
            src_lengths,
            return_all_hiddens,
            padding_mask=padding_mask,
            features_only=features_only,
        )
        x = out["encoder_out"][0]
        encoder_padding_mask = None
        if len(out["encoder_padding_mask"]) > 0:
            encoder_padding_mask = out["encoder_padding_mask"][0]

        def cal_text_layers(x, padding_mask, return_all_hiddens=False):
            encoder_states = []
            for layer in self.text_encoder_layers:
                x = layer(x, padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)
            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)
            return x, encoder_states

        x, encoder_states = cal_text_layers(x, encoder_padding_mask, return_all_hiddens)
        if features_only:
            return {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [encoder_padding_mask]
                if encoder_padding_mask is not None
                else [],  # B x T
                "encoder_embedding": [],  # B x T x C
                "encoder_states": encoder_states,  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [],
            }

        x_u = out["encoder_unmasked_out"][0]
        x_u, _ = cal_text_layers(x_u, encoder_padding_mask)

        return {
            "encoder_out": [x],  # [T x B x C]
            "encoder_unmasked_out": [x_u],  # [T x B x C]
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask is not None
            else [],  # B x T
            "encoder_embedding": [],  #
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "mask_indices": out["mask_indices"],  # B X T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.speech_encoder.reorder_encoder_out(encoder_out, new_order)


@register_model("s2t_wav_transformer")
class S2TWavTransformerModel(S2TTransformerModel):
    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        SpeechWavTransformerEncoder.add_args(parser)

    @classmethod
    def build_encoder(cls, args):
        encoder = SpeechWavTransformerEncoder(args)
        # Load pretrained wav2vec
        if args.load_pretrained_wav2vec_encoder:
            encoder.laod_pretrained_wav2vec(args.load_pretrained_wav2vec_encoder)
        # add CTC weight
        encoder.ctc_proj = None
        if getattr(args, "ctc_weight", 0.0) > 0.0:
            encoder.ctc_proj = nn.Linear(args.encoder_embed_dim, args.tgt_dict_size)
        return encoder


@register_model_architecture(
    "s2t_wav_transformer", "s2t_wav_transformer_base"
)
def speech_text_pretrain_bart_base(args):
    # speech masking
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)
    args.speech_mask_length = getattr(args, "speech_mask_length", 0)
    args.speech_mask_prob = getattr(args, "speech_mask_prob", 0.)
    args.speech_sup_mask_prob = getattr(args, "speech_sup_mask_prob", 0.)
    args.speech_unsup_mask_prob = getattr(
        args, "speech_unsup_mask_prob", args.speech_mask_prob
    )
    args.speech_mask_selection = getattr(args, "speech_mask_selection", "static")
    args.speech_mask_other = getattr(args, "speech_mask_other", 0)
    args.speech_mask_min_space = getattr(args, "speech_mask_min_space", 0)
    args.speech_no_mask_overlap = getattr(args, "speech_no_mask_overlap", False)

    args.speech_mask_channel_length = getattr(args, "speech_mask_channel_length", 0)
    args.speech_mask_channel_prob = getattr(args, "speech_mask_channel_prob", 0.0)
    args.speech_mask_channel_selection = getattr(
        args, "speech_mask_channel_selection", "static"
    )
    args.speech_mask_channel_other = getattr(args, "speech_mask_channel_other", 0)
    args.speech_mask_channel_min_space = getattr(
        args, "speech_mask_channel_min_space", 0
    )
    args.speech_no_mask_channel_overlap = getattr(
        args, "speech_no_mask_channel_overlap", False
    )
    args.no_scale_feature = getattr(args, "", False)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", .0)  # 0.1

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", args.encoder_embed_dim * 4
    )
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.speech_conv_bias = getattr(args, "speech_conv_bias", False)

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
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")  # gelu?
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)

    args.speech_unsup_dropout = getattr(args, "speech_unsup_dropout", 0)
    args.speech_unsup_feature_dropout = getattr(args, "speech_unsup_feature_dropout", 0)

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

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
