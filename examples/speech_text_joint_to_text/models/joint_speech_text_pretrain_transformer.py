#!/usr/bin/env python3

import logging
from collections import OrderedDict, namedtuple
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import checkpoint_utils, utils
from fairseq.file_io import PathManager
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import (
    MultiInputDecoder,
    MultiModalityEncoder,
    SpeechWavTransformerEncoder,
    StackedSpeechWavTransformerEncoder,
)
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
)

logger = logging.getLogger(__name__)


class SpeechTextPreTrainEncoder(MultiModalityEncoder):
    def __init__(
        self,
        dictionary,
        sup_speech_encoder,
        sup_s2s_speech_encoder,
        unsup_speech_encoder,
        text_encoder,
    ):
        super().__init__(dictionary)
        self.sup_speech_encoder = sup_speech_encoder
        self.sup_s2s_speech_encoder = sup_s2s_speech_encoder
        self.unsup_speech_encoder = unsup_speech_encoder
        self.text_encoder = text_encoder

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
            args, {"encoder_layers": args.text_encoder_layers}
        )
        text_encoder = TransformerEncoder(model_args, src_dictionary, enc_emb)
        return text_encoder

    @classmethod
    def build_speech_encoder(cls, args):
        model_args = cls.update_transformer_encoder_cfg(
            args,
            {
                "encoder_layers": args.speech_encoder_layers,
                "speech_mask_prob": args.speech_sup_mask_prob,
            },
        )
        speech_encoder = SpeechWavTransformerEncoder(model_args)
        return speech_encoder

    @classmethod
    def share_layers(cls, src_layers, tgt_layers):  # share layer but not dropout
        # share parameters in src_layers with tgt_layers
        assert len(src_layers) == len(tgt_layers)
        for i, ly in enumerate(src_layers):
            tly = tgt_layers[i]
            tly.self_attn = ly.self_attn
            tly.self_attn_layer_norm = ly.self_attn_layer_norm
            tly.activation_fn = ly.activation_fn
            tly.normalize_before = ly.normalize_before
            tly.fc1 = ly.fc1
            tly.fc2 = ly.fc2
            tly.final_layer_norm = ly.final_layer_norm
            if hasattr(tly, "encoder_attn"):
                tly.encoder_attn = ly.encoder_attn
                tly.encoder_attn_layer_norm = ly.encoder_attn_layer_norm
        return tgt_layers

    @classmethod
    def build_unsup_speech_encoder(cls, args, sup_speech_encoder):
        model_args = cls.update_transformer_encoder_cfg(
            args,
            {
                "encoder_layers": args.speech_encoder_layers,
                "speech_mask_prob": args.speech_unsup_mask_prob,
                "encoder_layerdrop": 0.0,
                "decoder_layerdrop": 0.0,
                "dropout": args.speech_unsup_dropout,
                "activation_dropout": args.speech_unsup_dropout,
                "attention_dropout": 0.0,
                "dropout_features": args.speech_unsup_feature_dropout,
                "dropout_input": args.speech_unsup_feature_dropout,
            },
        )

        unsup_speech_encoder = SpeechWavTransformerEncoder(model_args, alway_mask=True)
        unsup_speech_encoder.layer_norm = sup_speech_encoder.layer_norm
        unsup_speech_encoder.layers = cls.share_layers(
            sup_speech_encoder.layers, unsup_speech_encoder.layers
        )
        unsup_speech_encoder.mask_emb = sup_speech_encoder.mask_emb
        unsup_speech_encoder.embed_positions = sup_speech_encoder.embed_positions
        unsup_speech_encoder.feat_layer_norm = sup_speech_encoder.feat_layer_norm
        unsup_speech_encoder.feat_proj = sup_speech_encoder.feat_proj
        unsup_speech_encoder.subsample = sup_speech_encoder.subsample
        return unsup_speech_encoder

    @classmethod
    def build_encoder(cls, args, dictionary):
        text_encoder = cls.build_text_encoder(args, dictionary)
        if getattr(args, "load_pretrained_mbart_encoder_from", None):
            text_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=text_encoder,
                checkpoint=args.load_pretrained_mbart_encoder_from,
            )
        speech_encoder = cls.build_speech_encoder(args)
        if getattr(args, "load_pretrained_feature_extractor_from", None):

            def load_feature_extractor(component, checkpoint):
                if not PathManager.exists(checkpoint):
                    raise IOError("Model file not found: {}".format(checkpoint))
                state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
                component_state_dict = OrderedDict()

                component_prefix = "feature_extractor"
                for key in state["model"].keys():
                    if key.startswith(component_prefix):
                        component_subkey = key[len(component_prefix) + 1 :]
                        component_state_dict[component_subkey] = state["model"][key]
                component.load_state_dict(component_state_dict, strict=True)
                return component

            speech_encoder.subsample = load_feature_extractor(
                speech_encoder.subsample, args.load_pretrained_feature_extractor_from
            )
        speech_s2s_encoder = speech_encoder
        unsup_speech_encoder = cls.build_unsup_speech_encoder(args, speech_encoder)
        if getattr(args, "stacked_encoder", "none") != "none":
            if args.encoder_shared_text_layers_from_begin > 0:
                raise ValueError(
                    "We can not stack encoders and share encoders at the same time!"
                )
            speech_s2s_encoder = StackedSpeechWavTransformerEncoder(
                speech_encoder, text_encoder.layers, text_encoder.layer_norm
            )
            if args.stacked_encoder == "all":
                speech_encoder = speech_s2s_encoder
                unsup_speech_encoder = StackedSpeechWavTransformerEncoder(
                    unsup_speech_encoder, text_encoder.layers, text_encoder.layer_norm
                )
        else:
            cls.share_speech_text_encoder(
                speech_encoder, text_encoder, args.encoder_shared_text_layers_from_begin
            )
        return SpeechTextPreTrainEncoder(
            dictionary,
            speech_encoder,
            speech_s2s_encoder,
            unsup_speech_encoder,
            text_encoder,
        )

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

    def select_encoder(self, mode, **kwargs):
        if mode in ("speech", "sup_speech_ctc", "sup_speech_ali", "sup_speech_s2s"):
            kwargs["features_only"] = True
            if mode == "sup_speech_s2s":
                return self.sup_s2s_speech_encoder, kwargs
            return self.sup_speech_encoder, kwargs
        elif mode == "unsup_speech":
            kwargs["features_only"] = False
            return self.unsup_speech_encoder, kwargs
        elif mode in ("text", "bitext"):
            return self.text_encoder, kwargs
        else:
            raise NotImplementedError(f"{mode} is not supported")
        return None, kwargs

    def forward(self, src_tokens, src_lengths=None, mode="", alignment=None, **kwargs):
        return super().forward(src_tokens, src_lengths, mode, **kwargs)


# SpeechDummyDecoder works as an extension of encoder, so we could fit encoder only training into seq2seq training
class SpeechDummyDecoder(FairseqDecoder):
    def __init__(
        self,
        dictionary,
        output_embedding,
        no_emb_update_unsup=False,
        use_output_proj=False,
    ):
        super().__init__(dictionary)
        self.output_embedding = output_embedding
        num_embedding, num_dim = self.output_embedding.weight.size()
        self.out_proj = (
            None if use_output_proj is False else nn.Linear(num_dim, num_dim)
        )
        self.no_emb_update_unsup = no_emb_update_unsup

    def extend_alignment(self, alignment, src_lengths, prev_output_tokens):
        # alignment:    B X N
        # src_lengths:  B X T
        # prev_output_tokens:    B X (N + 1)
        tgt_tokens = prev_output_tokens[
            :, 1:
        ]  # remove the leading start of sentence token
        ext_alignment = (
            torch.ones(len(src_lengths), src_lengths.max(), device=src_lengths.device)
            .long()
            .fill_(self.dictionary.pad())
        )
        for bs in range(src_lengths.size(0)):
            tgt_length = tgt_tokens[bs].ne(self.dictionary.pad()).sum().item()
            assert tgt_length == sum(alignment[bs].ne(1)) + 1
            src_st = 0
            for i in range(tgt_length):
                tok = tgt_tokens[bs][i]
                src_ed = (alignment[bs][i] * src_lengths[bs]).int().item()
                ext_alignment[bs][src_st:src_ed].fill_(tok)
                src_st = src_ed
        return ext_alignment

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        mode="speech",
        alignment=None,
        **kwargs,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            sup_speech_ctc:
                dictionary{"logits": logits, "padding_mask": padding_mask}
            sup_speech_ali and unsup_speech:
                tuple:
                    - the decoder's output of shape `(batch, tgt_len, vocab)`
                    - a dictionary with any model-specific outputs
        """
        emb_weight = self.output_embedding.weight
        if (
            mode == "unsup_speech" and self.no_emb_update_unsup
        ):  # no gradient for embedding here
            emb_weight = emb_weight.detach()
        enc_out = (
            encoder_out["encoder_out"][0]
            if self.out_proj is None
            else self.out_proj(encoder_out["encoder_out"][0])
        )
        logits = F.linear(enc_out, emb_weight, None).transpose(0, 1)  # B X T X C
        others = None
        if mode in (
            "speech",
            "sup_speech_ctc",
        ):  # speech data with label, do forcealignment
            if len(encoder_out["encoder_padding_mask"]) > 0:
                padding_mask = encoder_out["encoder_padding_mask"][0]
                logits = logits.masked_fill(padding_mask, float("-inf"))
            else:
                seq_len, bsz = encoder_out["encoder_out"][0].size()[:2]
                padding_mask = torch.zeros(
                    bsz, seq_len, device=encoder_out["encoder_out"][0].device
                ).bool()
            return {"x": logits, "padding_mask": padding_mask}
        elif mode == "sup_speech_ali":
            src_lengths = None
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_lengths = (1 - encoder_out["encoder_padding_mask"][0].long()).sum(
                    -1
                )
            else:
                seq_len, bsz = encoder_out["encoder_out"][0].size()[:2]
                src_lengths = (
                    torch.ones(bsz, device=encoder_out["encoder_out"][0].device).long()
                    * seq_len
                )
            assert alignment is not None
            alignment = self.extend_alignment(
                alignment, src_lengths, prev_output_tokens
            )
            others = {"pseudo_target_tokens": alignment}
        elif mode == "unsup_speech":
            enc_out_ori = (
                encoder_out["encoder_unmasked_out"][0]
                if self.out_proj is None
                else self.out_proj(encoder_out["encoder_unmasked_out"][0])
            )
            logits_ori = F.linear(enc_out_ori, emb_weight, None).transpose(0, 1)
            if len(encoder_out["encoder_padding_mask"]) > 0:
                encoder_padding_mask = encoder_out["encoder_padding_mask"][0]
                logits_ori = logits_ori.masked_fill(encoder_padding_mask, float("-inf"))
            pseudo_labels = utils.log_softmax(logits_ori, dim=-1)
            others = {
                "pseudo_target_logprobs": pseudo_labels,
                "padding_mask": encoder_out["encoder_padding_mask"],  # B X T
                "mask_indices": encoder_out[
                    "mask_indices"
                ],  # True for masked frames B X T
            }
        return logits, others

    def get_normalized_probs(
        self,
        net_output: Dict[str, Tensor],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        return self.get_normalized_probs_scriptable(
            (net_output["x"], None), log_probs, sample
        )


class SpeechTextPreTrainDecoder(MultiInputDecoder):
    def __init__(self, dictionary, speech_decoder, text_decoder):
        super().__init__(dictionary)
        self.speech_decoder = speech_decoder
        self.text_decoder = text_decoder

    def select_decoder(self, mode, **kwargs):
        if mode == "unsup_speech":
            kwargs["mode"] = mode
            return self.speech_decoder, kwargs
        if mode in ("text", "bitext"):
            return self.text_decoder, kwargs
        if mode in ("speech", "sup_speech_ctc", "sup_speech_ali"):
            kwargs["mode"] = mode
            return self.speech_decoder, kwargs
        if mode in ("speech", "sup_speech_s2s"):
            if "alignment" in kwargs:
                del kwargs["alignment"]
            return self.text_decoder, kwargs

        raise NotImplementedError(f"{mode} is not supported")
        return None, kwargs

    def get_normalized_probs(
        self,
        net_output,
        log_probs,
        sample=None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        if isinstance(net_output, dict):
            return self.speech_decoder.get_normalized_probs(
                net_output, log_probs, sample
            )
        return self.text_decoder.get_normalized_probs(net_output, log_probs, sample)

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
    def build_dummy_speech_decoder(cls, args, dictionary, dec_emb_share=None):
        dec_emb = (
            nn.Embedding(len(dictionary), args.decoder_embed_dim, dictionary.pad())
            if dec_emb_share is None
            else dec_emb_share
        )
        speech_decoder = SpeechDummyDecoder(
            dictionary,
            dec_emb,
            no_emb_update_unsup=getattr(args, "no_emb_update_unsup", False),
            use_output_proj=getattr(args, "use_decoder_output_proj", False),
        )
        return speech_decoder

    @classmethod
    def build_decoder(
        cls, args, text_dictionary, speech_dictionary, speech_output_embedding
    ):
        text_decoder = cls.build_text_decoder(args, text_dictionary)
        speech_decoder = cls.build_dummy_speech_decoder(
            args, speech_dictionary, speech_output_embedding
        )
        if getattr(args, "load_pretrained_mbart_decoder_from", None):
            text_decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=text_decoder,
                checkpoint=args.load_pretrained_mbart_decoder_from,
            )
        return SpeechTextPreTrainDecoder(text_dictionary, speech_decoder, text_decoder)


@register_model("speech_text_pretrain_bart")
class SpeechTextPreTrainModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, src_lang_ids=None, **kwargs
    ):
        if src_lang_ids is not None:
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, src_lang_ids=src_lang_ids, **kwargs
            )
        else:
            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def max_positions(self):
        return None  # it is provided in task

    def get_targets(self, sample, net_output):
        mode = sample["net_input"]["mode"]
        if mode == "unsup_speech":
            return {"target_logprobs": net_output[1]["pseudo_target_logprobs"]}
        if mode == "sup_speech_ali":
            return net_output[1]["pseudo_target_tokens"]
        return sample["target"]

    def get_normalized_probs(
        self,
        net_output,
        log_probs,
        sample=None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        SpeechWavTransformerEncoder.add_args(parser)
        parser.add_argument(
            "--speech-sup-mask-prob",
            type=float,
            help="probability of replacing a token with mask (sup-speech)",
        )
        parser.add_argument(
            "--speech-unsup-mask-prob",
            type=float,
            help="probability of replacing a token with mask (unsup-speech)",
        )
        parser.add_argument(
            "--load-pretrained-mbart-encoder-from",
            type=str,
            metavar="STR",
            help="model to take text encoder  weights from (for initialization)",
        )

        parser.add_argument(
            "--load-pretrained-mbart-decoder-from",
            type=str,
            metavar="STR",
            help="model to take text decoder  weights from (for initialization)",
        )

        parser.add_argument(
            "--load-pretrained-feature-extractor-from",
            type=str,
            metavar="STR",
            help="model to take feature extractor weights from (for initialization)",
        )

        parser.add_argument(
            "--speech-unsup-dropout",
            type=float,
            default=0,
            help="dropout for unsupervised speech encoder",
        )

        parser.add_argument(
            "--speech-unsup-feature-dropout",
            type=float,
            default=0,
            help="dropout for unsupervised speech feature encoder",
        )

        parser.add_argument(
            "--encoder-shared-text-layers-from-begin",
            type=int,
            help="number of text encoder layers shared with speech encoder (from first layer)",
        )

        parser.add_argument(
            "--stacked-encoder",
            default="none",
            choices=["none", "s2s", "all"],
            help="stack speech and text encoders",
        )

        parser.add_argument("--use-decoder-output-proj", action="store_true")

    @classmethod
    def build_model(cls, args, task):
        encoder = SpeechTextPreTrainEncoder.build_encoder(args, task.src_dict)
        decoder = SpeechTextPreTrainDecoder.build_decoder(
            args, task.tgt_dict, task.src_dict, encoder.text_encoder.embed_tokens
        )
        model = SpeechTextPreTrainModel(encoder, decoder)
        return model

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        if "decoder.speech_decoder.output_projection.weight" in state_dict:
            del state_dict["decoder.speech_decoder.output_projection.weight"]
        self.upgrade_state_dict_named(state_dict, "")


@register_model_architecture(
    "speech_text_pretrain_bart", "speech_text_pretrain_bart_base"
)
def speech_text_pretrain_bart_base(args):
    # speech masking
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)
    args.speech_mask_length = getattr(args, "speech_mask_length", 10)
    args.speech_mask_prob = getattr(args, "speech_mask_prob", 0.65)
    args.speech_sup_mask_prob = getattr(args, "speech_sup_mask_prob", 0.3)
    args.speech_unsup_mask_prob = getattr(
        args, "speech_unsup_mask_prob", args.speech_mask_prob
    )
    args.speech_mask_selection = getattr(args, "speech_mask_selection", "static")
    args.speech_mask_other = getattr(args, "speech_mask_other", 0)
    args.speech_mask_min_space = getattr(args, "speech_mask_min_space", 1)
    args.speech_no_mask_overlap = getattr(args, "speech_no_mask_overlap", False)

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
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)  # 0.1

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

    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 6
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    args.no_emb_update_unsup = getattr(args, "no_emb_update_unsup", False)


@register_model_architecture(
    "speech_text_pretrain_bart", "speech_text_pretrain_bart_base_stack"
)
def speech_text_pretrain_bart_base_stack(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 0
    )
    args.stacked_encoder = getattr(args, "stacked_encoder", "all")
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    speech_text_pretrain_bart_base(args)


@register_model_architecture(
    "speech_text_pretrain_bart", "speech_text_pretrain_bart_large"
)
def speech_text_pretrain_bart_large(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 24)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 12)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 12
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.dropout = getattr(args, "dropout", 0.3)
    speech_text_pretrain_bart_base(args)


@register_model_architecture(
    "speech_text_pretrain_bart", "speech_text_pretrain_bart_large_stack"
)
def speech_text_pretrain_bart_large_stack(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 12)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 0
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.stacked_encoder = getattr(args, "stacked_encoder", "s2s")
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    speech_text_pretrain_bart_base(args)
