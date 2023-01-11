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
from fairseq.models.speech_to_speech.modules.ctc_decoder import CTCDecoder
from fairseq.models.speech_to_speech.modules.transformer_encoder import (
    TransformerEncoderNoEmb,
)
from fairseq.models.speech_to_text.xm_transformer import XMTransformerModel
from fairseq.models.speech_to_text.xm_transformer import (
    base_architecture as xm_t_base_architecture,
)
from fairseq.models.speech_to_text.xm_transformer import (
    build_embedding,
    need_finetuning,
    set_default_adaptor_args,
    set_default_general_args,
    set_default_transformer_decoder_args,
    set_default_w2v_encoder_args,
)
from fairseq.models.transformer import Linear, TransformerDecoder, TransformerModelBase
from fairseq.models.transformer.transformer_decoder_aug import AugTransformerDecoder

logger = logging.getLogger(__name__)


def unit_transformer_decoder_arch_base(
    args, decoder_layers=6, decoder_embed_dim=768, decoder_attention_heads=12
):
    args.encoder_layers = decoder_layers
    args.decoder_layers = decoder_layers
    args.decoder_embed_dim = decoder_embed_dim
    args.decoder_ffn_embed_dim = decoder_embed_dim * 4
    args.decoder_attention_heads = decoder_attention_heads
    args.encoder_embed_dim = args.decoder_embed_dim
    args.decoder_output_dim = decoder_embed_dim
    args.decoder_input_dim = decoder_embed_dim


def unit_transformer_decoder_arch_large(
    args, decoder_layers=12, decoder_embed_dim=1024, decoder_attention_heads=16
):
    args.encoder_layers = decoder_layers
    args.decoder_layers = decoder_layers
    args.decoder_embed_dim = decoder_embed_dim
    args.decoder_ffn_embed_dim = decoder_embed_dim * 4
    args.decoder_attention_heads = decoder_attention_heads
    args.encoder_embed_dim = args.decoder_embed_dim
    args.decoder_output_dim = decoder_embed_dim
    args.decoder_input_dim = decoder_embed_dim


@register_model("unity_xm_transformer")
class XMTransformerModelUnitY(XMTransformerModel):
    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/s2t"
        model_ids = []
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        XMTransformerModel.add_args(parser)
        parser.add_argument(
            "--translation-decoder-layers",
            type=int,
            default=4,
            metavar="N",
            help="num decoder layers in the first-pass translation module",
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
        parser.add_argument(
            "--load-pretrained-aux-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )

    @classmethod
    def build_text_decoder(cls, args, tgt_dict):
        _args = copy.deepcopy(args)

        if args.adaptor_proj or args.encoder_proj:  # not V0 arch
            _args.encoder_embed_dim = _args.decoder_embed_dim
        _args.dropout = args.decoder_dropout
        _args.attention_dropout = args.decoder_attention_dropout
        _args.activation_dropout = args.decoder_activation_dropout
        _args.layerdrop = _args.decoder_layerdrop
        _args.decoder_layers = _args.translation_decoder_layers

        embed_tokens = build_embedding(tgt_dict, _args.decoder_embed_dim)
        decoder = TransformerDecoder(_args, tgt_dict, embed_tokens)

        if getattr(args, "load_pretrained_aux_decoder_from", None) is not None:
            decoder = cls.maybe_load_pretrained(
                decoder, getattr(args, "load_pretrained_aux_decoder_from", None)
            )

            for k, p in decoder.named_parameters():
                p.requires_grad = need_finetuning(args.finetune_decoder_params, k)
        return decoder

    @classmethod
    def build_decoder(cls, args, task, aug_attn=False):
        _args = copy.deepcopy(args)
        _args.layerdrop = 0.0  # turn off layerdrop for shallow layers

        _args.encoder_embed_dim = args.decoder_embed_dim

        proj = None
        if args.decoder_embed_dim != _args.decoder_embed_dim:
            proj = Linear(args.decoder_embed_dim, _args.decoder_embed_dim)

        embed_tokens = build_embedding(task.target_dictionary, _args.decoder_embed_dim)
        decoder_cls = AugTransformerDecoder if aug_attn else TransformerDecoder
        decoder = decoder_cls(_args, task.target_dictionary, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None) is not None:
            # load all layers first and then discard the bottom layers
            embed_tokens = build_embedding(
                task.target_dictionary, _args.decoder_embed_dim
            )
            decoder_tmp = decoder_cls(_args, task.target_dictionary, embed_tokens)
            decoder_tmp = cls.maybe_load_pretrained(
                decoder_tmp, getattr(_args, "load_pretrained_decoder_from", None)
            )
            state_dict = decoder_tmp.state_dict()
            for k, p in decoder.named_parameters():
                p.data = state_dict[k].data
                p.requires_grad = need_finetuning(_args.finetune_decoder_params, k)
            decoder.layers = decoder.layers[-_args.decoder_layers :]

        return decoder, proj, _args

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        xm_t_base_architecture(args)

        encoder = cls.build_encoder(args)
        decoder, proj, unit_args = cls.build_decoder(
            args,
            task,
            aug_attn=getattr(args, "synthesizer_augmented_cross_attention", False),
        )
        base_model = cls(encoder, decoder)
        setattr(base_model, "proj", proj)

        base_model.t2u_augmented_cross_attn = getattr(
            args, "synthesizer_augmented_cross_attention", False
        )

        # set up multitask decoders
        base_model.mt_task_name = None
        base_model.multitask_decoders = {}
        has_first_pass_decoder = False
        for task_name, task_obj in task.multitask_tasks.items():
            if task_obj.is_first_pass_decoder:
                has_first_pass_decoder = True
                base_model.mt_task_name = task_name

            task_decoder = cls.build_multitask_decoder(
                args,
                task_obj.args,
                task_obj.target_dictionary,
                args.decoder_embed_dim,
                task_obj.is_first_pass_decoder,
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

        assert has_first_pass_decoder, "set at least one intermediate non-CTC decoder"

        # set up encoder on top of the auxiliary MT decoder
        if getattr(args, "synthesizer_encoder_layers", 0) > 0:
            base_model.synthesizer_encoder = cls.build_t2u_encoder(unit_args)
        else:
            base_model.synthesizer_encoder = None

        return base_model

    @classmethod
    def build_t2u_encoder(cls, args):
        _args = copy.deepcopy(args)
        _args.encoder_layers = _args.synthesizer_encoder_layers
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
        return_all_hiddens=False,
        tgt_speaker=None,
        **kwargs,
    ):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths, **kwargs
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
        if self.proj is not None:
            x = self.proj(x)

        mt_decoder_padding_mask = None
        if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
            mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

        # 2. T2U encoder
        if self.synthesizer_encoder is not None:
            t2u_encoder_out = self.synthesizer_encoder(
                x,
                mt_decoder_padding_mask,
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
                encoder_out_aug=t2u_encoder_out,
            )
        else:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=t2u_encoder_out,
            )
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_out"]
            # NOTE: from the top layer
            decoder_out[-1]["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ]
        decoder_out[-1]["mt_decoder_out"] = mt_decoder_out
        return decoder_out


@register_model_architecture(
    model_name="unity_xm_transformer", arch_name="unity_xm_transformer"
)
def base_architecture_unity(args):
    set_default_general_args(args)
    set_default_w2v_encoder_args(args)
    set_default_adaptor_args(args)
    set_default_transformer_decoder_args(args)

    args.layernorm_embedding = False
    args.decoder_learned_pos = False


# for old models
@register_model_architecture(
    model_name="unity_xm_transformer", arch_name="xm_transformer_t2"
)
def base_architecture_unity_legacy(args):
    base_architecture_unity(args)
