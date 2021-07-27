# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from examples.simultaneous_translation.modules.monotonic_transformer_layer import (
    TransformerMonotonicDecoderLayer,
    TransformerMonotonicEncoderLayer,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
    tiny_architecture
)
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
READ_ACTION = 0
WRITE_ACTION = 1

TransformerMonotonicDecoderOut = NamedTuple(
    "TransformerMonotonicDecoderOut",
    [
        ("action", int),
        ("p_choose", Optional[Tensor]),
        ("attn_list", Optional[List[Optional[Dict[str, Tensor]]]]),
        ("encoder_out", Optional[Dict[str, List[Tensor]]]),
        ("encoder_padding_mask", Optional[Tensor]),
    ],
)


@register_model("transformer_unidirectional")
class TransformerUnidirectionalModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)


@register_model("transformer_monotonic")
class TransformerModelSimulTrans(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)


class TransformerMonotonicEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerMonotonicEncoderLayer(args)
                for i in range(args.encoder_layers)
            ]
        )


class TransformerMonotonicDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerMonotonicDecoderLayer(args)
                for _ in range(args.decoder_layers)
            ]
        )
        self.policy_criterion = getattr(args, "policy_criterion", "any")

    def pre_attention(
        self,
        prev_output_tokens,
        encoder_out_dict: Dict[str, List[Tensor]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = encoder_out_dict["encoder_out"][0]

        if "encoder_padding_mask" in encoder_out_dict:
            encoder_padding_mask = (
                encoder_out_dict["encoder_padding_mask"][0]
                if encoder_out_dict["encoder_padding_mask"]
                and len(encoder_out_dict["encoder_padding_mask"]) > 0
                else None
            )
        else:
            encoder_padding_mask = None

        return x, encoder_out, encoder_padding_mask

    def post_attention(self, x):
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

    def clean_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
    ):
        """
        Clean cache in the monotonic layers.
        The cache is generated because of a forward pass of decoder has run but no prediction,
        so that the self attention key value in decoder is written in the incremental state.
        end_id is the last idx of the layers
        """
        if end_id is None:
            end_id = len(self.layers)

        for index, layer in enumerate(self.layers):
            if index < end_id:
                layer.prune_incremental_state(incremental_state)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,  # unused
        alignment_layer: Optional[int] = None,  # unused
        alignment_heads: Optional[int] = None,  # unsed
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # incremental_state = None
        assert encoder_out is not None
        (x, encoder_outs, encoder_padding_mask) = self.pre_attention(
            prev_output_tokens, encoder_out, incremental_state
        )
        attn = None
        inner_states = [x]
        attn_list: List[Optional[Dict[str, Tensor]]] = []

        p_choose = torch.tensor([1.0])

        for i, layer in enumerate(self.layers):

            x, attn, _ = layer(
                x=x,
                encoder_out=encoder_outs,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None
                else None,
            )

            inner_states.append(x)
            attn_list.append(attn)

            if incremental_state is not None:
                if_online = incremental_state["online"]["only"]
                assert if_online is not None
                if if_online.to(torch.bool):
                    # Online indicates that the encoder states are still changing
                    assert attn is not None
                    if self.policy_criterion == "any":
                        # Any head decide to read than read
                        head_read = layer.encoder_attn._get_monotonic_buffer(incremental_state)["head_read"]
                        assert head_read is not None
                        if head_read.any():
                            # We need to prune the last self_attn saved_state
                            # if model decide not to read
                            # otherwise there will be duplicated saved_state
                            self.clean_cache(incremental_state, i + 1)

                            return x, TransformerMonotonicDecoderOut(
                                action=0,
                                p_choose=p_choose,
                                attn_list=None,
                                encoder_out=None,
                                encoder_padding_mask=None,
                            )

        x = self.post_attention(x)

        return x, TransformerMonotonicDecoderOut(
            action=1,
            p_choose=p_choose,
            attn_list=attn_list,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
        )


@register_model_architecture("transformer_monotonic", "transformer_monotonic")
def base_monotonic_architecture(args):
    base_architecture(args)
    args.encoder_unidirectional = getattr(args, "encoder_unidirectional", False)


@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_iwslt_de_en"
)
def transformer_monotonic_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
    base_monotonic_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_vaswani_wmt_en_de_big"
)
def transformer_monotonic_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_vaswani_wmt_en_fr_big"
)
def transformer_monotonic_vaswani_wmt_en_fr_big(args):
    transformer_monotonic_vaswani_wmt_en_fr_big(args)


@register_model_architecture(
    "transformer_unidirectional", "transformer_unidirectional_iwslt_de_en"
)
def transformer_unidirectional_iwslt_de_en(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("transformer_monotonic", "transformer_monotonic_tiny")
def monotonic_tiny_architecture(args):
    tiny_architecture(args)
    base_monotonic_architecture(args)
