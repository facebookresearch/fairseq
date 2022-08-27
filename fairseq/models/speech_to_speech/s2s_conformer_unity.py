# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, List, Optional

import torch.nn as nn
from torch import Tensor

from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_speech.modules import CTCDecoder
from fairseq.models.speech_to_speech.s2s_conformer import S2UTConformerModel
from fairseq.models.speech_to_speech.s2s_transformer import (
    TransformerUnitDecoder,
    base_multitask_text_transformer_decoder_arch,
    s2ut_architecture_base,
)
from fairseq.models.transformer import Linear, TransformerDecoder, TransformerModelBase
from fairseq.models.transformer.transformer_decoder_aug import AugTransformerDecoder
from fairseq.modules import LayerNorm, TransformerEncoderLayer

logger = logging.getLogger(__name__)


def multitask_text_transformer_decoder_arch(
    args, decoder_layers, decoder_embed_dim=256, decoder_attention_heads=4
):
    args.decoder_layers = decoder_layers
    args.decoder_embed_dim = decoder_embed_dim
    args.decoder_attention_heads = decoder_attention_heads
    base_multitask_text_transformer_decoder_arch(args)


@register_model("unity_conformer")
class UnitYConformerModel(S2UTConformerModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + MT Transformer decoder + Transformer discrete unit decoder (UnitY)
    """

    @staticmethod
    def add_args(parser):
        S2UTConformerModel.add_args(parser)
        parser.add_argument(
            "--translation-decoder-layers",
            type=int,
            default=4,
            metavar="N",
            help="num decoder layers in the first-pass translation module",
        )
        parser.add_argument(
            "--synthesizer",
            default="transformer",
            choices=["transformer"],
            help="",
        )
        parser.add_argument(
            "--synthesizer-encoder-layers",
            type=int,
            default=0,
            metavar="N",
            help="num encoder layers in the second-pass synthesizer module",
        )
        parser.add_argument(
            "--synthesizer-augmented-cross-attention",
            action="store_true",
            default=False,
            help="augmented cross-attention over speech encoder output",
        )

    @classmethod
    def build_multitask_decoder(
        cls,
        args,
        tgt_dict,
        in_dim,
        is_mt_decoder,
        decoder_layers,
        decoder_embed_dim,
        decoder_attention_heads,
    ):
        decoder_args = args.decoder_args
        decoder_args.encoder_embed_dim = in_dim
        if args.decoder_type == "transformer":
            if is_mt_decoder:
                multitask_text_transformer_decoder_arch(
                    decoder_args,
                    decoder_layers,
                    decoder_embed_dim,
                    decoder_attention_heads,
                )  # 4L
            else:
                base_multitask_text_transformer_decoder_arch(decoder_args)  # 2L
            task_decoder = TransformerDecoder(
                decoder_args,
                tgt_dict,
                embed_tokens=TransformerModelBase.build_embedding(
                    decoder_args,
                    tgt_dict,
                    decoder_args.decoder_embed_dim,
                ),
            )
        elif args.decoder_type == "ctc":
            task_decoder = CTCDecoder(
                dictionary=tgt_dict,
                in_dim=in_dim,
            )
        else:
            raise NotImplementedError(
                "currently only support multitask decoder_type 'transformer', 'ctc'"
            )

        return task_decoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, aug_attn=False):
        from fairseq.models.speech_to_speech.modules import StackedEmbedding

        num_embeddings = len(tgt_dict)
        padding_idx = tgt_dict.pad()
        embed_tokens = StackedEmbedding(
            num_embeddings,
            args.decoder_embed_dim,
            padding_idx,
            num_stacked=args.n_frames_per_step,
        )

        _args = copy.deepcopy(args)
        _args.encoder_embed_dim = args.decoder_embed_dim

        decoder_cls = AugTransformerUnitDecoder if aug_attn else TransformerUnitDecoder
        return decoder_cls(
            _args,
            tgt_dict,
            embed_tokens,
        )

    @classmethod
    def build_model(cls, args, task):
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(
            args,
            task.target_dictionary,
            aug_attn=getattr(args, "synthesizer_augmented_cross_attention", False),
        )
        base_model = cls(encoder, decoder)

        base_model.t2u_augmented_cross_attn = getattr(
            args, "synthesizer_augmented_cross_attention", False
        )

        # set up multitask decoders
        is_mt_decoder = False
        base_model.mt_task_name = None
        base_model.multitask_decoders = {}
        n_aux_tasks = len(list(task.multitask_tasks.items()))
        for i, (task_name, task_obj) in enumerate(task.multitask_tasks.items()):
            if i == n_aux_tasks - 1:
                is_mt_decoder = True
                base_model.mt_task_name = task_name
                assert "target" in task_name
                assert task_obj.args.decoder_type == "transformer"
                # NOTE: we assume that the last task is for the first-pass decoder

            in_dim = (
                args.encoder_embed_dim
                if task_obj.args.input_from == "encoder"
                else args.decoder_embed_dim
            )
            task_decoder = cls.build_multitask_decoder(
                task_obj.args,
                task_obj.target_dictionary,
                in_dim,
                is_mt_decoder,
                getattr(args, "translation_decoder_layers", 4),
                getattr(args, "decoder_embed_dim", 256),
                getattr(args, "decoder_attention_heads", 4),
            )

            setattr(base_model, f"{task_name}_decoder", task_decoder)
            decoder_model_cls = (
                FairseqEncoderModel
                if task_obj.args.decoder_type == "ctc"
                else FairseqLanguageModel
            )
            base_model.multitask_decoders[task_name] = decoder_model_cls(
                getattr(base_model, f"{task_name}_decoder")
            )

        assert is_mt_decoder, "set at least one intermediate non-CTC decoder"

        # set up encoder on top of the auxiliary MT decoder
        if getattr(args, "synthesizer_encoder_layers", 0) > 0:
            base_model.synthesizer_encoder = cls.build_text_encoder(args)
        else:
            base_model.synthesizer_encoder = None

        return base_model

    @classmethod
    def build_text_encoder(cls, args):
        _args = copy.deepcopy(args)
        _args.encoder_layers = args.synthesizer_encoder_layers
        _args.encoder_embed_dim = args.decoder_embed_dim
        _args.encoder_ffn_embed_dim = args.decoder_ffn_embed_dim
        _args.encoder_attention_heads = args.decoder_attention_heads
        _args.encoder_normalize_before = True
        return TransformerEncoderNoEmb(_args)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        prev_output_tokens_mt,
        tgt_speaker=None,
        return_all_hiddens=False,
    ):
        mt_decoder = getattr(self, f"{self.mt_task_name}_decoder")

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            tgt_speaker=tgt_speaker,
            return_all_hiddens=return_all_hiddens,
        )

        # 1. MT decoder
        mt_decoder_out = mt_decoder(
            prev_output_tokens_mt,
            encoder_out=encoder_out,
        )
        x = mt_decoder_out[1]["inner_states"][-1]
        if mt_decoder.layer_norm is not None:
            x = mt_decoder.layer_norm(x)

        mt_decoder_padding_mask = None
        if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
            mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

        # 2. T2U encoder
        if hasattr(self, "synthesizer_encoder"):
            t2u_encoder_out = self.synthesizer_encoder(
                x,
                mt_decoder_padding_mask,
                return_all_hiddens=return_all_hiddens,
            )
        else:
            t2u_encoder_out = {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [mt_decoder_padding_mask],  # B x T
            }

        # 3. T2U decoder
        if self.t2u_augmented_cross_attn:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                encoder_out2=t2u_encoder_out,
            )
        else:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=t2u_encoder_out,
            )
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
            decoder_out[-1]["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ]
        decoder_out[-1]["mt_decoder_out"] = mt_decoder_out
        return decoder_out


class TransformerEncoderNoEmb(FairseqEncoder):
    """Transformer encoder without token embeddings."""

    def __init__(self, args):
        super().__init__(None)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, encoder_padding_mask, return_all_hiddens=False):

        encoder_states = []

        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask is not None and encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
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


class AugTransformerUnitDecoder(AugTransformerDecoder):
    """Based on Transformer decoder, with support to decoding stacked units"""

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn, output_projection
        )
        self.n_frames_per_step = args.n_frames_per_step

        self.out_proj_n_frames = (
            Linear(
                self.output_embed_dim,
                self.output_embed_dim * self.n_frames_per_step,
                bias=False,
            )
            if self.n_frames_per_step > 1
            else None
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        encoder_out2: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            encoder_out2=encoder_out2,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            bsz, seq_len, d = x.size()
            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)
            x = self.output_layer(x.view(bsz, seq_len, self.n_frames_per_step, d))
            x = x.view(bsz, seq_len * self.n_frames_per_step, -1)
            if (
                incremental_state is None and self.n_frames_per_step > 1
            ):  # teacher-forcing mode in training
                x = x[
                    :, : -(self.n_frames_per_step - 1), :
                ]  # remove extra frames after <eos>

        return x, extra

    def upgrade_state_dict_named(self, state_dict, name):
        if self.n_frames_per_step > 1:
            move_keys = [
                (
                    f"{name}.project_in_dim.weight",
                    f"{name}.embed_tokens.project_in_dim.weight",
                )
            ]
            for from_k, to_k in move_keys:
                if from_k in state_dict and to_k not in state_dict:
                    state_dict[to_k] = state_dict[from_k]
                    del state_dict[from_k]


@register_model_architecture(model_name="unity_conformer", arch_name="unity_conformer")
def unity_conformer_architecture_base(args):
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.max_source_positions = getattr(args, "max_source_positions", 6000)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    s2ut_architecture_base(args)


# for old models
@register_model_architecture(
    model_name="unity_conformer", arch_name="s2ut_conformer_translatotron2"
)
def unity_conformer_architecture_base_legacy(args):
    unity_conformer_architecture_base(args)
