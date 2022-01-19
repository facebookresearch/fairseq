# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional
from pathlib import Path
import torch.nn as nn
from torch import Tensor
from fairseq import checkpoint_utils

from fairseq.models import register_model, register_model_architecture
from fairseq.utils import safe_hasattr
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    TransformerDecoderScriptable
)
from fairseq.models.speech_to_text.s2t_transformer import base_architecture as s2t_base_architecture

from ..modules.attn_head_selector import AttnHeadSelector
from ..modules.head_selection_transformer_layer import HeadSelectionTransformerEncoderLayer
from .head_selection_transformer import HeadSelectionTransformerDecoder


logger = logging.getLogger(__name__)


@register_model("head_selection_s2t_transformer")
class HeadSelectionS2TTransformerModel(S2TTransformerModel):
    """
    Head selection implemented in S2TTransformer
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        # encoder head selection
        parser.add_argument(
            "--encoder-attn-head-select",
            action="store_true",
            default=False,
            help="encoder head selection"
        )
        parser.add_argument(
            "--total-encoder-attention-heads",
            type=int,
            help="total number of encoder attention heads"
        )
        # parser.add_argument(
        #     "--encoder-tasks",
        #     type=int,
        #     help="the number of encoder tasks (input languages or input domains)"
        # )
        # decoder self attention
        parser.add_argument(
            "--decoder-self-attn-head-select",
            action="store_true",
            default=False,
            help="decoder self-attention head selection"
        )
        # decoder-encoder attention
        parser.add_argument(
            "--dec-enc-attn-head-select",
            action="store_true",
            default=False,
            help="decoder-encoder attention head selection"
        )
        parser.add_argument(
            "--total-decoder-attention-heads",
            type=int,
            help="total number of decoder attention heads"
        )
        # parser.add_argument(
        #     "--decoder-tasks",
        #     type=int,
        #     help="the number of decoder tasks (output languages or output domains)"
        # )
        # selection strategy
        parser.add_argument(
            "--attn-head-select-strategy",
            type=str,
            help="attention head selection strategy, subset or group"
        )

    @classmethod
    def build_encoder(cls, args):
        if safe_hasattr(args, "encoder_attn_head_select") and args.encoder_attn_head_select:
            encoder = HeadSelectionS2TTransformerEncoder(args)
        else:
            encoder = S2TTransformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        if (safe_hasattr(args, "decoder_self_attn_head_select") and args.decoder_self_attn_head_select) or (safe_hasattr(args, "dec_enc_attn_head_select") and args.dec_enc_attn_head_select):
            return HeadSelectionTransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)
        else:
            return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)


class HeadSelectionS2TTransformerEncoder(S2TTransformerEncoder):

    def __init__(self, args):
        super().__init__(args)
        self.attn_head_selector = AttnHeadSelector(
            args.encoder_tasks,
            args.encoder_layers,
            args.total_encoder_attention_heads,
            args.encoder_attention_heads,
            args.attn_head_select_strategy,
        )
        self.task_ids = None
        self.transformer_layers = nn.ModuleList([
            HeadSelectionTransformerEncoderLayer(args, layer_idx, attn_head_selector=self.attn_head_selector) for layer_idx in range(args.encoder_layers)
        ])

    def set_task_ids(self, task_ids):
        self.task_ids = task_ids

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        self.attn_head_selector.head_select(self.task_ids)
        return super()._forward(src_tokens, src_lengths, return_all_hiddens)


class HeadSelectionTransformerDecoderScriptable(HeadSelectionTransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


@register_model_architecture(model_name="head_selection_s2t_transformer", arch_name="head_selection_s2t_transformer")
def base_architecture(args):
    s2t_base_architecture(args)
    args.encoder_attn_head_select = getattr(args, "encoder_attn_head_select", False)
    args.decoder_self_attn_head_select = getattr(args, "decoder_self_attn_head_select", False)
    args.dec_enc_attn_head_select = getattr(args, "dec_enc_attn_head_select", False)
    args.total_encoder_attention_heads = getattr(args, "total_encoder_attention_heads", 8)
    args.total_decoder_attention_heads = getattr(args, "total_decoder_attention_heads", 8)
    args.attn_head_select_strategy = getattr(args, "attn_head_select_strategy", "group")


@register_model_architecture("head_selection_s2t_transformer", "head_selection_s2t_transformer_s")
def head_selection_s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)
