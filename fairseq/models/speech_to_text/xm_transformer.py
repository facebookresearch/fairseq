#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import copy
from typing import Dict, List, Optional, Tuple

from fairseq import utils, checkpoint_utils
from fairseq.models import (FairseqEncoderDecoderModel, FairseqEncoder,
                            register_model, register_model_architecture)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.wav2vec import Wav2VecEncoder
from fairseq.modules.layer_norm import LayerNorm
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.utils import safe_hasattr
from torch import Tensor
import torch.nn as nn


logger = logging.getLogger(__name__)


class Conv1dAdaptor(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=3, kernel_size=3, stride=2,
                 add_layernorm=False):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Conv1d(in_dim if i == 0 else out_dim, out_dim * 2, kernel_size,
                      stride=stride, padding=kernel_size // 2)
            for i in range(n_layers)
        )
        self.layernorms = None
        if add_layernorm:
            self.layernorms = nn.ModuleList(LayerNorm(out_dim)
                                            for _ in range(n_layers))
        self.stride = stride

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--adaptor-n-layers", type=int)
        parser.add_argument("--adaptor-kernel-size", type=int)
        parser.add_argument("--adaptor-stride", type=int)
        parser.add_argument("--adaptor-layernorm", action='store_true')

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in self.layers:
            out = ((out.float() - 1) / self.stride + 1).floor().long()
        return out

    def forward(self, x, padding_mask):
        # T x B x C -> B x C x T
        x = x.transpose(0, 1).transpose(1, 2)
        for i, layer in enumerate(self.layers):
            x = nn.functional.glu(layer(x), dim=1)
            if self.layernorms is not None:
                x = self.layernorms[i](x.transpose(1, 2)).transpose(1, 2)
        # B x C x T -> T x B x C
        x = x.transpose(1, 2).transpose(0, 1)

        if padding_mask is None:
            out_padding_mask = None
        else:
            out_lengths = self.get_out_seq_lens_tensor((~padding_mask).sum(1))
            out_padding_mask = lengths_to_padding_mask(out_lengths)
        return x, out_padding_mask


def add_wav2vec_asr_args(parser):
    parser.add_argument("--w2v-path", help="path to wav2vec 2.0 model")
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        help="if true, does not load pretrained weights",
    )
    parser.add_argument(
        "--dropout-input",
        type=float,
        metavar="D",
        help="dropout to apply to the input (after feat extr)",
    )
    parser.add_argument(
        "--final-dropout",
        type=float,
        metavar="D",
        help="dropout after transformer and before final projection",
    )
    parser.add_argument(
        "--apply-mask", action="store_true", help="apply masking during fine-tuning"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        metavar="D",
        help="dropout probability inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--activation-dropout",
        "--relu-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside wav2vec 2.0 model",
    )

    parser.add_argument(
        "--mask-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-prob", type=float, help="probability of replacing a token with mask"
    )

    parser.add_argument(
        "--mask-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--mask-channel-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-channel-prob",
        type=float,
        help="probability of replacing a token with mask",
    )

    parser.add_argument(
        "--mask-channel-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-channel-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-channel-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--freeze-finetune-updates",
        default=0,
        type=int,
        help="dont finetune wav2vec for this many updates",
    )

    parser.add_argument(
        "--feature-grad-mult",
        default=None,
        type=float,
        help="reset feature grad mult in wav2vec 2.0 to this",
    )

    parser.add_argument(
        "--layerdrop",
        default=0.0,
        type=float,
        help="probability of dropping a layer in wav2vec 2.0",
    )
    parser.add_argument("--w2v-args", default=None)


class Wav2VecEncoderWithAdaptor(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None)
        self.w2v_encoder = Wav2VecEncoder(args)
        encoder_out_dim = self.w2v_encoder.w2v_model.encoder.embedding_dim
        # Projection + 8x shrinking
        self.adaptor = Conv1dAdaptor(
            encoder_out_dim, args.decoder_embed_dim,
            n_layers=args.adaptor_n_layers,
            kernel_size=args.adaptor_kernel_size, stride=args.adaptor_stride,
            add_layernorm=args.adaptor_layernorm
        )
        for k, p in self.w2v_encoder.w2v_model.named_parameters():
            # Freeze pretrained models by default
            if safe_hasattr(args, 'finetune_w2v_params') and XMTransformerModel.finetune_params(
                    args.finetune_w2v_params, k):
                p.requires_grad = True
            else:
                p.requires_grad = False

    @classmethod
    def add_args(cls, parser):
        add_wav2vec_asr_args(parser)
        parser.add_argument(
            "--normalize", action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument("--finetune-w2v-params", type=str, metavar="STR",
                            help="comma-separated param strings to finetune.")
        Conv1dAdaptor.add_args(parser)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        padding_mask = lengths_to_padding_mask(src_lengths)
        out = self.w2v_encoder.forward(src_tokens, padding_mask, tbc=True)
        x = out["encoder_out"]
        enc_padding_mask = None
        if out["encoder_padding_mask"] is not None:
            enc_padding_mask = out["encoder_padding_mask"].transpose(0, 1)   # T X B --> B X T

        x, enc_padding_mask = self.adaptor(x, enc_padding_mask)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [enc_padding_mask] if enc_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in
                  encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in
                  encoder_out["encoder_embedding"]]
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


def add_decoder_args(parser):
    parser.add_argument("--activation-fn", type=str, default='relu',
                        choices=utils.get_available_activation_fns(),
                        help="activation function to use")
    parser.add_argument("--decoder-dropout", type=float, metavar="D",
                        help="dropout probability")
    parser.add_argument("--decoder-attention-dropout", type=float,
                        metavar="D",
                        help="dropout probability for attention weights")
    parser.add_argument("--decoder-activation-dropout", type=float,
                        metavar="D",
                        help="dropout probability after activation in FFN.")
    parser.add_argument("--decoder-embed-dim", type=int, metavar="N",
                        help="decoder embedding dimension")
    parser.add_argument("--decoder-ffn-embed-dim", type=int, metavar="N",
                        help="decoder embedding dimension for FFN")
    parser.add_argument("--decoder-layers", type=int, metavar="N",
                        help="num decoder layers")
    parser.add_argument("--decoder-attention-heads", type=int, metavar="N",
                        help="num decoder attention heads")
    parser.add_argument("--decoder-normalize-before", action="store_true",
                        help="apply layernorm before each decoder block")
    parser.add_argument("--layernorm-embedding", action="store_true",
                        help="add layernorm to embedding")
    parser.add_argument("--no-scale-embedding", action="store_true",
                        help="if True, dont scale embeddings")
    parser.add_argument(
        "--load-pretrained-decoder-from", type=str, metavar="STR",
        help="model to take decoder weights from (for initialization)"
    )
    parser.add_argument("--finetune-decoder-params", type=str,
                        metavar="STR",
                        help="comma-separated param strings to finetune.")
    parser.add_argument("--checkpoint-activations", action="store_true")


@register_model("xm_transformer")
class XMTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        Wav2VecEncoderWithAdaptor.add_args(parser)
        add_decoder_args(parser)

    @classmethod
    def build_encoder(cls, args):
        _args = copy.deepcopy(args)
        state = checkpoint_utils.load_checkpoint_to_cpu(args.w2v_path)
        if state.get("cfg") is not None:
            encoder_embed_dim = state["cfg"]._content["model"]["encoder_embed_dim"]
        elif state.get("args") is not None:
            encoder_embed_dim = state["args"].encoder_embed_dim
        else:
            raise ValueError(f"Invalid config in {args.w2v_path}")
        _args.decoder_embed_dim = encoder_embed_dim
        encoder = Wav2VecEncoderWithAdaptor(_args)
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        _args = copy.deepcopy(args)
        _args.dropout = args.decoder_dropout
        _args.attention_dropout = args.decoder_attention_dropout
        _args.activation_dropout = args.decoder_activation_dropout
        _args.max_target_positions = 1024

        decoder = TransformerDecoder(_args, task.target_dictionary,
                                     embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        for k, p in decoder.named_parameters():
            # Freeze pretrained models by default
            if safe_hasattr(args, 'finetune_decoder_params') and XMTransformerModel.finetune_params(
                    args.finetune_decoder_params, k):
                p.requires_grad = True
            else:
                p.requires_grad = False
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(task.target_dictionary,
                                               args.decoder_embed_dim)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs,
                                                      sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens,
                                   src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens,
                                   encoder_out=encoder_out)
        return decoder_out

    def upgrade_state_dict(self, state_dict):
        for k, _ in state_dict.items():
            if 'adaptor.layers' in state_dict:
                print(k)
                new = k.replace('adaptor.layers', 'adaptor_layers')
                state_dict[new] = state_dict[k]
                del state_dict[k]

    @staticmethod
    def finetune_params(finetune_params, param_name):
        if finetune_params == "all":
            return True
        finetune_params_list = finetune_params.split(",")
        for finetune_param in finetune_params_list:
            if finetune_param in param_name:
                return True
        return False


def set_default_w2v_encoder_args(args):
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_before = getattr(args, "mask_channel_before", False)
    args.mask_channel_selection = getattr(args, "mask_channel_selection",
                                          "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap",
                                           False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = 0.1
    args.layerdrop = getattr(args, "layerdrop", 0.0)

    args.normalize = getattr(args, "normalize", False)


def set_default_adaptor_args(args):
    args.adaptor_n_layers = getattr(args, "adaptor_n_layers", 3)
    args.adaptor_kernel_size = getattr(args, "adaptor_kernel_size", 3)
    args.adaptor_stride = getattr(args, "adaptor_stride", 2)
    args.adaptor_layernorm = getattr(args, "adaptor_layernorm", False)


def set_default_mbart_decoder_args(args):
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim',
                                         4 * 1024)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before',
                                            True)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_attention_dropout = getattr(args, 'decoder_attention_dropout',
                                             0.)
    args.decoder_activation_dropout = getattr(args,
                                              'decoder_activation_dropout', 0.)
    args.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff',
                                           None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_output_dim = getattr(args, 'decoder_output_dim',
                                      args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim',
                                     args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)


@register_model_architecture(model_name="xm_transformer",
                             arch_name="xm_transformer")
def base_architecture(args):
    set_default_w2v_encoder_args(args)
    set_default_adaptor_args(args)
    set_default_mbart_decoder_args(args)
