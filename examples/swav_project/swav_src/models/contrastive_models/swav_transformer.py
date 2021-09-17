from fairseq.models.bart.model import BARTModel
from typing import Optional
import argparse

import torch
import torch.nn as nn
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from fairseq.models.transformer import DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS, \
    DEFAULT_MIN_PARAMS_TO_WRAP, TransformerModel
from fairseq.models import transformer
from fairseq.models.roberta import RobertaModel
from fairseq.models import roberta
from fairseq.models import bart

# MASS XLM CODE implementation
from ..xlm_code_transformers.mass_transformer import MassTransformerFairseqModel
from ..xlm_code_transformers.mass_transformer import MassTransformerDecoder, \
    MassTransformerFairseqEncoderModel, MassTransformerEncoder
from ..xlm_code_transformers import mass_transformer
from ..xlm_code_transformers.xlm_code_transformer import XLMCodeTransformerDecoder, XLMCodeTransformerEncoder, \
    XLMCodeTransformerFairseqEncoderModel, XLMCodeTransformerFairseqModel
from ..xlm_code_transformers import xlm_code_transformer
from ...swav_utils.utils import check_num_stability

import logging

logger = logging.getLogger(__name__)

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def xlm_bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def upgrade_state_dict_named_prototypes(model, state_dict, name):
    if "prototypes.weight" not in state_dict:
        prot_state = model.prototypes.state_dict()
        for k, v in prot_state.items():
            state_dict[f'prototypes.{k}'] = v


def universal_build_prototypes(args: argparse.Namespace):
    nmb_prototypes = getattr(args, 'nmb_prototypes', -1)
    if isinstance(nmb_prototypes, list):
        prototypes = MultiPrototypes(args.encoder_embed_dim, nmb_prototypes)
    elif nmb_prototypes > 0:
        prototypes = nn.Linear(args.encoder_embed_dim, nmb_prototypes, bias=False)
    return prototypes


def xlm_universal_build_prototypes(args):
    nmb_prototypes = getattr(args, 'nmb_prototypes', -1)
    if isinstance(nmb_prototypes, list):
        prototypes = MultiPrototypes(args.emb_dim, nmb_prototypes)
    elif nmb_prototypes > 0:
        prototypes = nn.Linear(args.emb_dim, nmb_prototypes, bias=False)
    return prototypes


def add_swav_args(parser):
    parser.add_argument("--nmb-prototypes", default=3000, type=int,
                        help="number of prototypes")
    parser.add_argument("--freeze-prototypes-niters", default=313, type=int,
                        help="freeze the prototypes during this many iterations from the start")


@register_model("swav_transformer")
class SwavTransformer(TransformerModel):
    def __init__(self, args, encoder, decoder, prototypes):
        super().__init__(args, encoder, decoder)
        self.prototypes = prototypes
        self.num_updates = None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def add_args(parser):
        TransformerModel.add_args(parser=parser)
        add_swav_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        Difference from TransformerModel:
        Add prototype layers
        """

        # make sure all arguments are present in older models
        transformer.base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)

        prototypes = cls.build_prototypes(args)
        return cls(args, encoder, decoder, prototypes)

    @classmethod
    def build_prototypes(cls, args):
        return universal_build_prototypes(args)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: torch.Tensor,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        get_prototypes: bool = False,
        get_prototypes_only: bool = False,
        pre_norm_prototypes: bool = True,
        **kwargs,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.

        Extra args:
            get_prototypes: return original output with prototype output into extra dict {'prot_out', 'prot_embed'}
            get_prototypes_only: only return prot extra dict, skip the decoder or other parts
                beware of unused params error of decoder and encoder projection params
            pre_norm_prototypes: turn on pre-normalization of prototype layer (should be True)
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)
        prot_extra = {}
        x_feature = encoder_out['encoder_out'][0]
        if get_prototypes or get_prototypes_only:
            # NOTE: left_pad_source must be turn OFF, roberta: OFF by default, translation: maybe not
            _embed = x_feature[0]  # take <s> token (equiv. to [CLS])
            prot_out = self.prototypes(_embed)
            prot_embed = _embed.detach()
            prot_extra['prot_out'] = prot_out
            prot_extra['prot_embed'] = prot_embed
            if get_prototypes_only:
                # NOTE: Don't use for training, this may cause decoder params to be unused,
                #   enable --find-unsued-parameters if need to
                return {
                    'prot_out': prot_out, 
                    'prot_embed': prot_embed
                }
        else:
            # add dummy prototypes usaged to prevent error of unsued-parameters
            x_feature += self.prototypes(x_feature.new(1, x_feature.size(-1)).fill_(0))[0, 0] * 0
            encoder_out['encoder_out'] = [x_feature]

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        x, extra = decoder_out
        for k, v in prot_extra.items():
            extra[k] = v
        decoder_out = x, extra
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        upgrade_state_dict_named_prototypes(self, state_dict, name)


@register_model_architecture("swav_transformer", "swav_transformer_tiny")
def tiny_swav_transformer(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.rand_factor = getattr(args, "rand_factor", 2)
    return transformer.base_architecture(args)


@register_model("swav_bart")
class SwavBARTModel(BARTModel):
    def __init__(self, args, encoder, decoder, prototypes):
        super().__init__(args, encoder, decoder)
        self.prototypes = prototypes
        self.num_updates = None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def add_args(parser):
        BARTModel.add_args(parser=parser)
        add_swav_args(parser)
        parser.add_argument("--prot-hidden", default="bos", type=str,
                            help="prot-hidden")

    def extract_prot_hidden(self, src_tokens, src_lengths, encoder_output, **kwargs):
        """Encoder output: [T, B, D] encoder_out['encoder_out'][0]"""
        # 1: left_pad_source must be turn OFF, as it may not for seq2seq model
        #   for TransformerEncoder encoder_out: [T, B, D]
        prot_hidden = getattr(self.args, 'prot_hidden', 'bos')
        if prot_hidden == "bos":
            return encoder_output[0]  # take <s> token (equiv. to [CLS])
        elif prot_hidden == "eos":
            _x_prot_feed = encoder_output.transpose(0, 1)
            mask = torch.arange(_x_prot_feed.size(1)).to(_x_prot_feed).unsqueeze_(0) == (src_lengths - 1).unsqueeze_(-1)
            sentence_representation = _x_prot_feed[mask, :]
            return sentence_representation
        elif prot_hidden == "avg_pooling":
            has_langtok = kwargs.get('has_langtok', True)
            _x_prot_feed = encoder_output.transpose(0, 1)
            mask = torch.arange(_x_prot_feed.size(1)).to(_x_prot_feed).unsqueeze_(0) < (src_lengths - 1).unsqueeze_(-1)
            mask = mask.unsqueeze_(-1)
            sent_embeds = _x_prot_feed * mask
            if has_langtok:
                mask = mask[:, 1:]
                sent_embeds = sent_embeds[:, 1:]
            sent_embed = (sent_embeds / mask.sum(1, keepdim=True)).sum(1)
            return sent_embed
        else:
            raise ValueError(f'prot_hidden: {self.args.prot_hidden} not found')

    @classmethod
    def build_prototypes(cls, args):
        return SwavTransformer.build_prototypes(args)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        upgrade_state_dict_named_prototypes(self, state_dict, name)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        transformer.base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)

        prototypes = cls.build_prototypes(args)
        return cls(args, encoder, decoder, prototypes)

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: Optional[torch.Tensor] = None,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        get_prototypes: bool = False, 
        get_prototypes_only: bool = False,
        pre_norm_prototypes: bool = False,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens
        )
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)
        prot_extra = {}
        eos: int = self.eos
        x_feature = encoder_out['encoder_out'][0]
        if get_prototypes or get_prototypes_only:
            # # 1: left_pad_source must be turn OFF, it is turn OFF by default for roberta
            # # for TransformerEncoder encoder_out: [T, B, D]
            _embed = self.extract_prot_hidden(src_tokens, src_lengths, x_feature)
            prot_embed = _embed.detach()

            # prot_out = prot_embed
            prot_out = self.prototypes(_embed)

            assert prot_out.dim() == 2, f'prot_out dim not 2 {prot_out.size()}, {_embed.size()}'
            assert not torch.any(torch.isinf(prot_out))
            prot_extra['prot_out'] = prot_out
            prot_extra['prot_embed'] = prot_embed
            if get_prototypes_only:
                # NOTE: WARNING this will cause decoder params to be unused!
                return {
                    'prot_out': prot_out,
                    'prot_embed': prot_embed
                }
        else:
            # add dummy prototypes to preven error
            x_feature += self.prototypes(x_feature.new(1, x_feature.size(-1)).fill_(0))[0, 0] * 0
            encoder_out['encoder_out'] = [x_feature]
        if prev_output_tokens is None:
            assert get_prototypes or get_prototypes_only
            return x_feature, prot_extra

        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        for k, v in prot_extra.items():
            extra[k] = v

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(eos), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break
        return x, extra


@register_model_architecture("swav_bart", "swav_bart_large")
def swav_bart_large_architecture(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    bart.model.bart_large_architecture(args)


@register_model_architecture("swav_bart", "swav_mbart_large")
def swav_mbart_large_architecture(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    bart.model.mbart_large_architecture(args)


@register_model("swav_mass_transformer")
class SwavMassTransformer(MassTransformerFairseqModel):
    def __init__(self, args, encoder, decoder, prototypes):
        super().__init__(args, encoder, decoder)
        self.prototypes = prototypes
        self.num_updates = None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @staticmethod
    def add_args(parser):
        MassTransformerFairseqModel.add_args(parser=parser)
        add_swav_args(parser)

    @classmethod
    def build_prototypes(cls, args):
        prototypes = SwavMassEncoderModel.build_prototypes(args)
        return prototypes

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        upgrade_state_dict_named_prototypes(self, state_dict, name)

    @classmethod
    def build_model(cls, args, task):
        # make sure all arguments are present in older models
        mass_transformer.base_architecture(args)
        encoder = MassTransformerEncoder(args, task.source_dictionary, with_output=cls.encoder_with_output())
        decoder = MassTransformerDecoder(args, task.target_dictionary)

        prototypes = cls.build_prototypes(args)
        return cls(args, encoder, decoder, prototypes)

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        get_prototypes: bool = False, 
        get_prototypes_only: bool = False,
        pre_norm_prototypes: bool = False,
        **kwargs,
    ):
        """
        NOTE: SwavMassTransformer must train swav loss in combination w/ MASS loss

        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.

        Extra args:
            get_prototypes: return original output with prototype output into extra dict {'prot_out', 'prot_embed'}
            get_prototypes_only: only return prot extra dict, skip the decoder or other parts
                beware of unused params error of decoder and encoder projection params
            pre_norm_prototypes: turn on pre-normalization of prototype layer
        """
        if alignment_layer is not None or alignment_heads is not None:
            raise NotImplementedError('alignment not impl.')
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

        # infer src_langs
        src_langs = self.infer_langs(src_tokens, "src", kwargs, 0)
        tgt_langs = self.infer_langs(prev_output_tokens, "tgt", kwargs, 1)
        assert src_langs is not None
        kwargs.pop("src_langs", None)
        kwargs.pop("tgt_langs", None)

        # encoder
        x = self.encoder(
            src_tokens, src_lengths,
            features_only=True, return_all_hiddens=True,
            src_langs=src_langs, in_model_forward=True, **kwargs
        )
        src_enc = x
        x_feature = x
        src_len = src_lengths

        # prototypes output
        prot_extra = {}
        if get_prototypes or get_prototypes_only:
            _embed = x_feature[:, 0]  # take <s> token (equiv. to [CLS])
            assert torch.all(src_tokens[:, 0] == self.encoder.eos_index), f'beginning not eos {src_tokens}'
            prot_out = self.prototypes(_embed)
            prot_embed = _embed.detach()
            assert prot_out.dim() == 2, f'prot_out dim not 2 {prot_out.size()}, {_embed.size()}'
            assert not torch.any(torch.isinf(prot_out))
            prot_extra['prot_out'] = prot_out
            prot_extra['prot_embed'] = prot_embed
            if get_prototypes_only:
                # NOTE: WARNING this will cause decoder params to be unused!
                return {
                    'prot_out': prot_out, 
                    'prot_embed': prot_embed
                }
        else:
            # add dummy prototypes to prevent error
            src_enc += self.prototypes(x_feature.new(1, x_feature.size(-1)).fill_(0))[0, 0] * 0
        if prev_output_tokens is None:
            assert get_prototypes or get_prototypes_only
            return x_feature, prot_extra

        # decoder (should match super-class)
        assert tgt_langs is not None
        tgt_len = kwargs.get("tgt_lengths", (prev_output_tokens != self.decoder.pad_index).int().sum(-1))

        dec_out = self.decoder(
            prev_output_tokens, tgt_len, 
            src_enc=src_enc, src_len=src_len, src_langs=tgt_langs, 
            features_only=False, in_model_forward=True
        )
        out, extra = dec_out
        for k, v in prot_extra.items(): 
            extra[k] = v
        dec_out = out, extra
        return dec_out


@register_model_architecture("swav_mass_transformer", "swav_mass_transformer_big")
def swav_mass_transformer_big(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    return mass_transformer.mass_transformer_big(args)


@register_model("swavednc_mass_transformer")
class SwavEncDecMassTransformer(SwavMassTransformer):
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        get_prototypes: bool = False, 
        get_prototypes_only: bool = False,
        pre_norm_prototypes: bool = False,
        **kwargs,
    ):
        """
        NOTE: SwavEncDecMassTransformer must train swav loss in combination w/ MASS loss

        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.

        Extra args:
            get_prototypes: return original output with prototype output into extra dict {'prot_out', 'prot_embed'}
            get_prototypes_only: only return prot extra dict, skip the decoder or other parts
                beware of unused params error of decoder and encoder projection params
            pre_norm_prototypes: turn on pre-normalization of prototype layer
        """
        if alignment_layer is not None or alignment_heads is not None:
            raise NotImplementedError('alignment not impl.')
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

        # infer src_langs
        src_langs = self.infer_langs(src_tokens, "src", kwargs, 0)
        tgt_langs = self.infer_langs(prev_output_tokens, "tgt", kwargs, 1)
        assert src_langs is not None
        kwargs.pop("src_langs", None)
        kwargs.pop("tgt_langs", None)

        # encoder
        x = self.encoder(
            src_tokens, src_lengths, 
            features_only=True, return_all_hiddens=True,
            src_langs=src_langs, in_model_forward=True, **kwargs
        )
        src_enc = x
        x_feature = x
        src_len = src_lengths
        enc_tensor = x_feature
        # prototypes output
        prot_extra = {}
        assert prev_output_tokens is not None

        # decoder (should match super-class)
        assert tgt_langs is not None
        tgt_len = kwargs.get("tgt_lengths", (prev_output_tokens != self.decoder.pad_index).int().sum(-1))

        dec_out = self.decoder(
            prev_output_tokens, tgt_len,
            src_enc=src_enc, src_len=src_len, src_langs=tgt_langs,
            features_only=True,     # change to True
            in_model_forward=True
        )
        dec_tensor, extra = dec_out

        if get_prototypes:
            # enc: take [:, 0] <s> token
            # dec: take [:, last_eos] <s/> token
            enc_embed = enc_tensor[:, 0]    # take <s> token
            assert torch.all(src_tokens[:, 0] == self.encoder.eos_index), f'beginning not eos {src_tokens}'
            end_points = torch.arange(dec_tensor.size(1), device=dec_tensor.device).unsqueeze_(0) == (tgt_len - 1).unsqueeze_(1)
            dec_embed = dec_tensor[end_points, :].view(
                dec_tensor.size(0), -1, dec_tensor.size(-1))[:, -1, :]
            prot_out = self.prototypes(enc_embed)
            prot_embed = enc_embed.detach()
            prot_extra['prot_out'] = prot_out
            prot_extra['prot_embed'] = prot_embed

            dec_prot_out = self.prototypes(dec_embed)
            dec_prot_embed = dec_embed.detach()
            check_num_stability('dec_prot_out', dec_prot_out)
            check_num_stability('dec_prot_embed', dec_prot_embed)
            
            prot_extra['dec_prot_out'] = dec_prot_out
            prot_extra['dec_prot_embed'] = dec_prot_embed
            if get_prototypes_only:
                return prot_extra

        out = self.decoder.compute_output(dec_tensor)
        extra = {**extra, **prot_extra}
        dec_out = out, extra
        return dec_out


@register_model_architecture("swavednc_mass_transformer", "swavednc_mass_transformer_big")
def swavednc_mass_transformer_big(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    return mass_transformer.mass_transformer_big(args)


@register_model_architecture("swavednc_mass_transformer", "swavednc_mass_transformer_big_p500")
def swavednc_mass_transformer_big_p500(args):
    args.nmb_prototypes = getattr(args, "nmb_prototypes", 500)
    return swav_mass_transformer_big(args)


@register_model("swav_roberta")
class SwavRobertaModel(RobertaModel):
    """
    To be used for roberta model
    """
    def __init__(self, args, encoder, prototypes):
        super().__init__(args, encoder)
        self.prototypes = prototypes
        self.num_updates = None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @staticmethod
    def add_args(parser):
        RobertaModel.add_args(parser=parser)
        add_swav_args(parser)
    
    @classmethod
    def build_prototypes(cls, args):
        prototypes = SwavTransformer.build_prototypes(args)
        return prototypes
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        roberta.model.base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        
        encoder = roberta.RobertaEncoder(args, task.source_dictionary)
        prototypes = cls.build_prototypes(args)
        return cls(args, encoder, prototypes)

    def forward(
        self,
        src_tokens: torch.Tensor,
        features_only: bool = False,
        return_all_hiddens: bool = False,
        classification_head_name: Optional[str] = None,
        get_prototypes: bool = False, 
        get_prototypes_only: bool = False,
        pre_norm_prototypes: bool = False,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True
        
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

        x_feature, extra = self.encoder(src_tokens, True, return_all_hiddens, **kwargs)

        if get_prototypes or get_prototypes_only:
            # 1: left_pad_source must be turn OFF, it is turn OFF by default for roberta
            #   RobertaEncoder [B, T, D]
            _embed = x_feature[:, 0]  # take <s> token (equiv. to [CLS])

            prot_out = self.prototypes(_embed)
            prot_embed = _embed.detach()
            assert prot_out.dim() == 2, f'prot_out dim not 2 {prot_out.size()}, {_embed.size()}'
            assert not torch.any(torch.isinf(prot_out))
            # NOTE must add dummy with encoder.output_layer to prevent error
            extra['prot_out'] = prot_out
            extra['prot_out'] += self.encoder.output_layer(prot_embed[:1])[0, 0] * 0
            extra['prot_embed'] = prot_embed
            if get_prototypes_only:
                return {
                    'prot_out': prot_out, 
                    'prot_embed': prot_embed,
                }
        
        x = x_feature if features_only else self.encoder.output_layer(
            x_feature, masked_tokens=kwargs.get('masked_tokens', None))

        # NOTE must add dummy with prototypes to prevent error
        x += self.prototypes(x_feature[:1, 0])[0, 0] * 0

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        upgrade_state_dict_named_prototypes(self, state_dict, name)


@register_model_architecture("swav_roberta", "swav_roberta_tiny")
def swav_roberta_tiny(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.rand_factor = getattr(args, "rand_factor", 2)
    return roberta.base_architecture(args)


@register_model_architecture("swav_roberta", "swav_roberta_base")
def swav_roberta_base(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    return roberta.roberta_base_architecture(args)


@register_model_architecture("swav_roberta", "swav_roberta_large")
def swav_roberta_large(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    return roberta.roberta_large_architecture(args)


@register_model_architecture("swav_roberta", "swav_roberta_xlm")
def swav_roberta_xlm(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    return roberta.xlm_architecture(args)


@register_model("swav_mass_encoder")
class SwavMassEncoderModel(MassTransformerFairseqEncoderModel):
    def __init__(self, args, encoder, prototypes):
        super().__init__(args, encoder)
        self.prototypes = prototypes
        self.num_updates = None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        # add prototypes to state_dict so that it can be loaded
        upgrade_state_dict_named_prototypes(self, state_dict, name)

    @staticmethod
    def add_args(parser):
        MassTransformerFairseqEncoderModel.add_args(parser=parser)
        add_swav_args(parser)
    
    @classmethod
    def build_prototypes(cls, args):
        return xlm_universal_build_prototypes(args)
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        mass_transformer.base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        
        # encoder = roberta.RobertaEncoder(args, task.source_dictionary)
        encoder = MassTransformerEncoder(args, task.source_dictionary)
        prototypes = cls.build_prototypes(args)
        return cls(args, encoder, prototypes)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        queue=None, 
        get_prototypes=False, 
        get_prototypes_only=False, 
        pre_norm_prototypes=False,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True
        
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

        x_feature = self.encoder(
            src_tokens, features_only=True, return_all_hiddens=return_all_hiddens, 
            in_model_forward=True, **kwargs)
        extra = {}

        if get_prototypes or get_prototypes_only:
            # 1: left_pad_source must be True, for roberta, it is turn on by default
            _embed = x_feature[:, 0]  # take <s> token (equiv. to [CLS])
            prot_out = self.prototypes(_embed)
            prot_embed = _embed.detach()
            assert prot_out.dim() == 2, f'prot_out dim not 2 {prot_out.size()}, {_embed.size()}'
            # NOTE must add dummy with encoder.output_layer to prevent error
            extra['prot_out'] = prot_out
            extra['prot_out'] = prot_out + self.encoder.compute_output(prot_embed[:1])[0, 0] * 0
            extra['prot_embed'] = prot_embed
            if get_prototypes_only:
                return {
                    'prot_out': prot_out, 
                    'prot_embed': prot_embed
                }
        
        x = x_feature if features_only else self.encoder.compute_output(
            x_feature, masked_tokens=kwargs.get('masked_tokens', None))

        # NOTE must add dummy with prototypes to prevent error
        x += self.prototypes(x_feature[:1, 0])[0, 0] * 0

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra


@register_model_architecture("swav_mass_encoder", "swav_mass_encoder_big")
def swav_mass_encoder_big(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    return mass_transformer.mass_encoder_big(args)


@register_model_architecture("swav_mass_encoder", "swav_mass_encoder_big_p500")
def swav_mass_encoder_big_p500(args):
    args.nmb_prototypes = getattr(args, "nmb_prototypes", 500)
    return swav_mass_encoder_big(args)


@register_model("swav_xlm_encoder")
class SwavXLMEncoderModel(XLMCodeTransformerFairseqEncoderModel):
    def __init__(self, args, encoder, prototypes):
        super().__init__(args, encoder)
        self.prototypes = prototypes
        self.num_updates = None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        upgrade_state_dict_named_prototypes(self, state_dict, name)

    @staticmethod
    def add_args(parser):
        XLMCodeTransformerFairseqEncoderModel.add_args(parser=parser)
        add_swav_args(parser)
    
    @classmethod
    def build_prototypes(cls, args):
        return xlm_universal_build_prototypes(args)
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        mass_transformer.base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        
        encoder = XLMCodeTransformerEncoder(args, task.source_dictionary)
        prototypes = cls.build_prototypes(args)
        return cls(args, encoder, prototypes)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        get_prototypes=False, 
        get_prototypes_only=False, 
        pre_norm_prototypes=False,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True
        
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

        x_feature = self.encoder(
            src_tokens, features_only=True, return_all_hiddens=return_all_hiddens, 
            in_model_forward=True,
            **kwargs
        )
        extra = {}

        if get_prototypes or get_prototypes_only:
            # 1: left_pad_source must be True, for roberta, it is turn on by default
            _embed = x_feature[:, 0]  # take <s> token (equiv. to [CLS])
            prot_out = self.prototypes(_embed)
            prot_embed = _embed.detach()
            assert prot_out.dim() == 2, f'prot_out dim not 2 {prot_out.size()}, {_embed.size()}'
            # NOTE must add dummy with encoder.output_layer to prevent error
            extra['prot_out'] = prot_out + self.encoder.compute_output(prot_embed[:1])[0, 0] * 0
            extra['prot_embed'] = prot_embed
            if get_prototypes_only:
                return {
                    'prot_out': prot_out, 
                    'prot_embed': prot_embed
                }
        
        x = x_feature if features_only else self.encoder.compute_output(
            x_feature, masked_tokens=kwargs.get('masked_tokens', None))

        # NOTE must add dummy with prototypes to prevent error
        x += self.prototypes(x_feature[:1, 0])[0, 0] * 0

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra


@register_model_architecture("swav_xlm_encoder", "swav_xlm_encoder_big")
def swav_xlm_encoder_big(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    return xlm_code_transformer.xlm_transformer_encoder_big(args)


@register_model_architecture("swav_xlm_encoder", "swav_xlm_encoder_big_p500")
def swav_xlm_encoder_big_p500(args):
    args.nmb_prototypes = getattr(args, "nmb_prototypes", 500)
    return swav_xlm_encoder_big(args)


@register_model("swav_xlm_transformer")
class SwavXLMTransformer(XLMCodeTransformerFairseqModel):
    def __init__(self, args, encoder, decoder, prototypes):
        super().__init__(args, encoder, decoder)
        self.prototypes = prototypes
        self.num_updates = None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @staticmethod
    def add_args(parser):
        XLMCodeTransformerFairseqModel.add_args(parser=parser)
        add_swav_args(parser)

    @classmethod
    def build_prototypes(cls, args):
        return xlm_universal_build_prototypes(args)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        upgrade_state_dict_named_prototypes(self, state_dict, name)

    @classmethod
    def build_model(cls, args, task):
        # make sure all arguments are present in older models
        xlm_code_transformer.base_architecture(args)
        encoder = XLMCodeTransformerEncoder(args, task.source_dictionary, with_output=cls.encoder_with_output())
        decoder = XLMCodeTransformerDecoder(args, task.target_dictionary)

        prototypes = cls.build_prototypes(args)
        return cls(args, encoder, decoder, prototypes)

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        get_prototypes: bool = False, 
        get_prototypes_only: bool = False,
        pre_norm_prototypes: bool = False,
        **kwargs,
    ):
        """
        NOTE: SwavXLMTransformer must train swav loss in combination w/ MASS loss

        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.

        Extra args:
            get_prototypes: return original output with prototype output into extra dict {'prot_out', 'prot_embed'}
            get_prototypes_only: only return prot extra dict, skip the decoder or other parts
                beware of unused params error of decoder and encoder projection params
            pre_norm_prototypes: turn on pre-normalization of prototype layer
        """
        if alignment_layer is not None or alignment_heads is not None:
            raise NotImplementedError('alignment not impl.')
        
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

        # infer src_langs
        src_langs = self.infer_langs(src_tokens, "src", kwargs, 0)
        tgt_langs = self.infer_langs(prev_output_tokens, "tgt", kwargs, 1)
        assert src_langs is not None
        kwargs.pop("src_langs", None)
        kwargs.pop("tgt_langs", None)

        # encoder
        x = self.encoder(
            src_tokens, src_lengths, 
            features_only=True, return_all_hiddens=True,
            src_langs=src_langs, in_model_forward=True, **kwargs
        )
        src_enc = x
        x_feature = x
        src_len = src_lengths

        # prototypes output
        prot_extra = {}
        if get_prototypes or get_prototypes_only:
            _embed = x_feature[:, 0]  # take <s> token (equiv. to [CLS])
            assert torch.all(src_tokens[:, 0] == self.encoder.eos_index), f'beginning not eos {src_tokens}'
            prot_out = self.prototypes(_embed)
            prot_embed = _embed.detach()
            assert prot_out.dim() == 2, f'prot_out dim not 2 {prot_out.size()}, {_embed.size()}'
            assert not torch.any(torch.isinf(prot_out))
            prot_extra['prot_out'] = prot_out
            prot_extra['prot_embed'] = prot_embed
            if get_prototypes_only:
                # NOTE: WARNING this will cause decoder params to be unused!
                return {
                    'prot_out': prot_out, 
                    'prot_embed': prot_embed
                }
        else:
            # add dummy prototypes to prevent error
            src_enc += self.prototypes(x_feature.new(1, x_feature.size(-1)).fill_(0))[0, 0] * 0
        if prev_output_tokens is None:
            assert get_prototypes or get_prototypes_only
            return x_feature, prot_extra

        # decoder (should match super-class)
        assert tgt_langs is not None
        tgt_len = kwargs.get("tgt_lengths", (prev_output_tokens != self.decoder.pad_index).int().sum(-1))

        out, extra = self.decoder(
            prev_output_tokens, tgt_len, 
            src_enc=src_enc, src_len=src_len, src_langs=tgt_langs, 
            features_only=False, in_model_forward=True
        )
        for k, v in prot_extra.items(): 
            extra[k] = v
        return out, extra


@register_model_architecture("swav_xlm_transformer", "swav_xlm_transformer_big")
def swav_xlm_transformer_big(args):
    args.rand_factor = getattr(args, "rand_factor", 2)
    return xlm_code_transformer.xlm_transformer_big(args)


@register_model_architecture("swav_xlm_transformer", "swav_xlm_transformer_big_p500")
def swav_xlm_transformer_big_p500(args):
    args.nmb_prototypes = getattr(args, "nmb_prototypes", 500)
    return swav_xlm_transformer_big(args)
