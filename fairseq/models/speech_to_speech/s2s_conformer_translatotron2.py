# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

from fairseq.models import (
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_speech.modules import CTCDecoder
from fairseq.models.speech_to_speech.s2s_conformer import TranslatotronConformerModel
from fairseq.models.speech_to_speech.s2s_conformer_unity import (
    TransformerEncoderNoEmb,
    multitask_text_transformer_decoder_arch,
)
from fairseq.models.speech_to_speech.s2s_transformer import (
    base_multitask_text_transformer_decoder_arch,
    translatotron_architecture_base,
)
from fairseq.models.text_to_speech import TTSTransformerDecoder
from fairseq.models.transformer import Linear, TransformerDecoder, TransformerModelBase

logger = logging.getLogger(__name__)


@register_model("translatotron2_conformer")
class Translatotron2ConformerModel(TranslatotronConformerModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + MT Transformer decoder + TTS Transformer decoder (Translatotron2)
    """

    @staticmethod
    def add_args(parser):
        TranslatotronConformerModel.add_args(parser)
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
    def build_decoder(cls, args):
        _args = copy.deepcopy(args)
        _args.encoder_embed_dim = args.decoder_embed_dim

        if args.synthesizer == "transformer":
            return TTSTransformerDecoder(_args, None, padding_idx=1)
        else:
            raise NotImplementedError(args.synthesizer)

    @classmethod
    def build_model(cls, args, task):
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args)
        base_model = cls(encoder, decoder)

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
        incremental_state=None,
        target_lengths=None,
        speaker=None,
        return_all_hiddens=False,
    ):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            tgt_speaker=tgt_speaker,
            return_all_hiddens=return_all_hiddens,
        )

        # 1. MT decoder
        mt_decoder = getattr(self, f"{self.mt_task_name}_decoder")
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

        # 2. TTS encoder
        if hasattr(self, "synthesizer_encoder"):
            tts_encoder_out = self.synthesizer_encoder(
                x,
                mt_decoder_padding_mask,
                return_all_hiddens=return_all_hiddens,
            )
        else:
            tts_encoder_out = {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [mt_decoder_padding_mask],  # B x T
            }

        # 3. TTS decoder
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=tts_encoder_out,
            incremental_state=incremental_state,
            target_lengths=target_lengths,
            speaker=speaker,
        )
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
            decoder_out[-1]["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ]
        decoder_out[-1]["mt_decoder_out"] = mt_decoder_out
        return decoder_out


@register_model_architecture(
    model_name="translatotron2_conformer", arch_name="translatotron2_conformer"
)
def translatotron2_conformer_architecture_base(args):
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.max_source_positions = getattr(args, "max_source_positions", 6000)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    translatotron_architecture_base(args)


# for old models
@register_model_architecture(
    model_name="translatotron2_conformer", arch_name="s2spect_conformer_translatotron2"
)
def translatotron2_conformer_architecture_base_legacy(args):
    translatotron2_conformer_architecture_base(args)
