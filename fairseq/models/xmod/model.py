# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ..roberta.model_xlmr import XLMRModel
from fairseq.models.xmod.transformer_layer_xmod import XMODTransformerEncoderLayerBase
from ..roberta.model import base_architecture, RobertaEncoder
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Optional
from fairseq.models.xmod.hub_interface import XMODHubInterface
import torch
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from fairseq.modules.checkpoint_activations import checkpoint_wrapper

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


@register_model("xmod")
class XMODModel(XLMRModel):
    @classmethod
    def hub_models(cls):
        return {
            "xmod.base": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.81.1M.tar.gz",
            "xmod.large.prenorm": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.large.prenorm.81.500k.tar.gz",
            "xmod.base.13.125k": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.13.125k.tar.gz",
            "xmod.base.30.125k": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.30.125k.tar.gz",
            "xmod.base.30.195k": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.30.195k.tar.gz",
            "xmod.base.60.125k": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.60.125k.tar.gz",
            "xmod.base.60.265k": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.60.265k.tar.gz",
            "xmod.base.75.125k": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.75.125k.tar.gz",
            "xmod.base.75.269k": "https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.75.269k.tar.gz",
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="sentencepiece",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return XMODHubInterface(x["args"], x["task"], x["models"][0])

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            if not hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample

        encoder = XMODEncoder(args, task.source_dictionary)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        lang_id=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True
        x, extra = self.encoder(
            src_tokens, features_only, return_all_hiddens, lang_id=lang_id, **kwargs
        )

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra


class XMODEncoder(RobertaEncoder):
    """XMOD encoder."""

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = XMODTransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        lang_id=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens, lang_id=lang_id
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(
        self, src_tokens, return_all_hiddens=False, lang_id=None, **kwargs
    ):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            lang_id=lang_id,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states}


class XMODTransformerEncoder(TransformerEncoder):
    def build_encoder_layer(self, cfg):
        layer = XMODTransformerEncoderLayerBase(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        lang_id=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens,
            src_lengths,
            return_all_hiddens,
            token_embeddings,
            lang_id=lang_id,
        )
        # TorchScript doesn't support super() method so that the scriptable Subclass
        # can't access the base class model in Torchscript.
        # Current workaround is to add a helper function with different name and
        # call the helper function from scriptable Subclass.

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        lang_id=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                lang_id=lang_id,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }


@register_model_architecture("xmod", "xmod_base_13")
def roberta_base_architecture(args):
    args.ffn_modules = getattr(args, "ffn_modules", False)
    args.adapter_modules = getattr(args, "adapter_modules", True)
    args.adapter_layer_norm = getattr(args, "adapter_layer_norm", False)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", True)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", True)
    args.languages = getattr(
        args,
        "languages",
        [
            "ar_AR",
            "en_XX",
            "fi_FI",
            "fr_XX",
            "hi_IN",
            "id_ID",
            "ka_GE",
            "ko_KR",
            "ru_RU",
            "sw_KE",
            "ta_IN",
            "th_TH",
            "vi_VN",
        ],
    )
    base_architecture(args)


@register_model_architecture("xmod", "xmod_base_30")
def roberta_base_architecture(args):
    args.ffn_modules = getattr(args, "ffn_modules", False)
    args.adapter_modules = getattr(args, "adapter_modules", True)
    args.adapter_layer_norm = getattr(args, "adapter_layer_norm", False)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", True)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", True)
    args.languages = getattr(
        args,
        "languages",
        [
            "ar_AR",
            "cs_CZ",
            "en_XX",
            "eu_ES",
            "fi_FI",
            "fr_XX",
            "hi_IN",
            "hr_HR",
            "hu_HU",
            "hy_AM",
            "id_ID",
            "it_IT",
            "ka_GE",
            "ko_KR",
            "lt_LT",
            "ml_IN",
            "mn_MN",
            "ms_MY",
            "pl_PL",
            "ro_RO",
            "ru_RU",
            "si_LK",
            "sk_SK",
            "sq_AL",
            "sv_SE",
            "sw_KE",
            "ta_IN",
            "th_TH",
            "tl_XX",
            "vi_VN",
        ],
    )
    base_architecture(args)


@register_model_architecture("xmod", "xmod_base_60")
def roberta_base_architecture(args):
    args.ffn_modules = getattr(args, "ffn_modules", False)
    args.adapter_modules = getattr(args, "adapter_modules", True)
    args.adapter_layer_norm = getattr(args, "adapter_layer_norm", False)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", True)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", True)
    args.languages = getattr(
        args,
        "languages",
        [
            "af_ZA",
            "am_ET",
            "ar_AR",
            "be_BY",
            "bn_IN",
            "ca_ES",
            "cs_CZ",
            "cy_GB",
            "da_DK",
            "en_XX",
            "eo_EO",
            "et_EE",
            "eu_ES",
            "fa_IR",
            "fi_FI",
            "fr_XX",
            "ga_IE",
            "gl_ES",
            "gu_IN",
            "ha_NG",
            "hi_IN",
            "hr_HR",
            "hu_HU",
            "hy_AM",
            "id_ID",
            "is_IS",
            "it_IT",
            "ka_GE",
            "ko_KR",
            "ku_TR",
            "la_VA",
            "lt_LT",
            "lv_LV",
            "mk_MK",
            "ml_IN",
            "mn_MN",
            "ms_MY",
            "ne_NP",
            "nl_XX",
            "no_XX",
            "pl_PL",
            "ps_AF",
            "pt_XX",
            "ro_RO",
            "ru_RU",
            "sa_IN",
            "sd_PK",
            "si_LK",
            "sk_SK",
            "sl_SI",
            "so_SO",
            "sq_AL",
            "sr_RS",
            "sv_SE",
            "sw_KE",
            "ta_IN",
            "te_IN",
            "th_TH",
            "tl_XX",
            "vi_VN",
        ],
    )
    base_architecture(args)


@register_model_architecture("xmod", "xmod_base_75")
def roberta_base_architecture(args):
    args.ffn_modules = getattr(args, "ffn_modules", False)
    args.adapter_modules = getattr(args, "adapter_modules", True)
    args.adapter_layer_norm = getattr(args, "adapter_layer_norm", False)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", True)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", True)
    args.languages = getattr(
        args,
        "languages",
        [
            "af_ZA",
            "am_ET",
            "ar_AR",
            "as_IN",
            "be_BY",
            "bn_IN",
            "br_FR",
            "bs_BA",
            "ca_ES",
            "cs_CZ",
            "cy_GB",
            "da_DK",
            "en_XX",
            "eo_EO",
            "et_EE",
            "eu_ES",
            "fa_IR",
            "fi_FI",
            "fr_XX",
            "fy_NL",
            "ga_IE",
            "gd_GB",
            "gl_ES",
            "gu_IN",
            "ha_NG",
            "hi_IN",
            "hr_HR",
            "hu_HU",
            "hy_AM",
            "id_ID",
            "is_IS",
            "it_IT",
            "jv_ID",
            "ka_GE",
            "kn_IN",
            "ko_KR",
            "ku_TR",
            "la_VA",
            "lt_LT",
            "lv_LV",
            "mg_MG",
            "mk_MK",
            "ml_IN",
            "mn_MN",
            "mr_IN",
            "ms_MY",
            "ne_NP",
            "nl_XX",
            "no_XX",
            "om_KE",
            "or_IN",
            "pa_IN",
            "pl_PL",
            "ps_AF",
            "pt_XX",
            "ro_RO",
            "ru_RU",
            "sa_IN",
            "sd_PK",
            "si_LK",
            "sk_SK",
            "sl_SI",
            "so_SO",
            "sq_AL",
            "sr_RS",
            "su_ID",
            "sv_SE",
            "sw_KE",
            "ta_IN",
            "te_IN",
            "th_TH",
            "tl_XX",
            "vi_VN",
            "xh_ZA",
            "yi_DE",
        ],
    )
    base_architecture(args)


@register_model_architecture("xmod", "xmod_base")
def roberta_base_architecture(args):
    args.ffn_modules = getattr(args, "ffn_modules", False)
    args.adapter_modules = getattr(args, "adapter_modules", True)
    args.adapter_layer_norm = getattr(args, "adapter_layer_norm", False)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", True)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", True)
    args.languages = getattr(
        args,
        "languages",
        [
            "en_XX",
            "id_ID",
            "vi_VN",
            "ru_RU",
            "fa_IR",
            "sv_SE",
            "ja_XX",
            "fr_XX",
            "de_DE",
            "ro_RO",
            "ko_KR",
            "hu_HU",
            "es_XX",
            "fi_FI",
            "uk_UA",
            "da_DK",
            "pt_XX",
            "no_XX",
            "th_TH",
            "pl_PL",
            "bg_BG",
            "nl_XX",
            "zh_CN",
            "he_IL",
            "el_GR",
            "it_IT",
            "sk_SK",
            "hr_HR",
            "tr_TR",
            "ar_AR",
            "cs_CZ",
            "lt_LT",
            "hi_IN",
            "zh_TW",
            "ca_ES",
            "ms_MY",
            "sl_SI",
            "lv_LV",
            "ta_IN",
            "bn_IN",
            "et_EE",
            "az_AZ",
            "sq_AL",
            "sr_RS",
            "kk_KZ",
            "ka_GE",
            "tl_XX",
            "ur_PK",
            "is_IS",
            "hy_AM",
            "ml_IN",
            "mk_MK",
            "be_BY",
            "la_VA",
            "te_IN",
            "eu_ES",
            "gl_ES",
            "mn_MN",
            "kn_IN",
            "ne_NP",
            "sw_KE",
            "si_LK",
            "mr_IN",
            "af_ZA",
            "gu_IN",
            "cy_GB",
            "eo_EO",
            "km_KH",
            "ky_KG",
            "uz_UZ",
            "ps_AF",
            "pa_IN",
            "ga_IE",
            "ha_NG",
            "am_ET",
            "lo_LA",
            "ku_TR",
            "so_SO",
            "my_MM",
            "or_IN",
            "sa_IN",
        ],
    )
    base_architecture(args)


@register_model_architecture("xmod", "xmod_large_prenorm")
def roberta_base_architecture(args):
    args.ffn_modules = getattr(args, "ffn_modules", False)
    args.adapter_modules = getattr(args, "adapter_modules", True)
    args.adapter_layer_norm = getattr(args, "adapter_layer_norm", True)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", False)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", False)
    # args.bottleneck = getattr(args, "bottleneck", 8)
    args.bottleneck = getattr(args, "bottleneck", 4)
    args.languages = getattr(
        args,
        "languages",
        [
            "en_XX",
            "id_ID",
            "vi_VN",
            "ru_RU",
            "fa_IR",
            "sv_SE",
            "ja_XX",
            "fr_XX",
            "de_DE",
            "ro_RO",
            "ko_KR",
            "hu_HU",
            "es_XX",
            "fi_FI",
            "uk_UA",
            "da_DK",
            "pt_XX",
            "no_XX",
            "th_TH",
            "pl_PL",
            "bg_BG",
            "nl_XX",
            "zh_CN",
            "he_IL",
            "el_GR",
            "it_IT",
            "sk_SK",
            "hr_HR",
            "tr_TR",
            "ar_AR",
            "cs_CZ",
            "lt_LT",
            "hi_IN",
            "zh_TW",
            "ca_ES",
            "ms_MY",
            "sl_SI",
            "lv_LV",
            "ta_IN",
            "bn_IN",
            "et_EE",
            "az_AZ",
            "sq_AL",
            "sr_RS",
            "kk_KZ",
            "ka_GE",
            "tl_XX",
            "ur_PK",
            "is_IS",
            "hy_AM",
            "ml_IN",
            "mk_MK",
            "be_BY",
            "la_VA",
            "te_IN",
            "eu_ES",
            "gl_ES",
            "mn_MN",
            "kn_IN",
            "ne_NP",
            "sw_KE",
            "si_LK",
            "mr_IN",
            "af_ZA",
            "gu_IN",
            "cy_GB",
            "eo_EO",
            "km_KH",
            "ky_KG",
            "uz_UZ",
            "ps_AF",
            "pa_IN",
            "ga_IE",
            "ha_NG",
            "am_ET",
            "lo_LA",
            "ku_TR",
            "so_SO",
            "my_MM",
            "or_IN",
            "sa_IN",
        ],
    )

    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)
