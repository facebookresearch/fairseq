# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Dict, Optional
import torch
import torch.nn as nn
from torch import Tensor

from fairseq.utils import safe_hasattr
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder
)

from ..modules.attn_head_selector import AttnHeadSelector
from ..modules.head_selection_transformer_layer import (
    HeadSelectionTransformerEncoderLayer,
    HeadSelectionTransformerDecoderLayer
)


class HeadSelectionTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
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
        # selection strategy
        parser.add_argument(
            "--attn-head-select-strategy",
            type=str,
            help="attention head selection strategy, subset or group"
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if safe_hasattr(args, "encoder_attn_head_select") and args.encoder_attn_head_select:
            return HeadSelectionTransformerEncoder(
                args, src_dict, embed_tokens
            )
        else:
            return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if (safe_hasattr(args, "decoder_self_attn_head_select") and args.decoder_self_attn_head_select) or (safe_hasattr(args, "dec_enc_attn_head_select") and args.dec_enc_attn_head_select):
            return HeadSelectionTransformerDecoder(
                args, tgt_dict, embed_tokens
            )
        else:
            return TransformerDecoder(args, tgt_dict, embed_tokens)


class HeadSelectionTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        self.num_tasks = args.encoder_tasks
        self.num_layers = args.encoder_layers
        self.total_num_heads = args.total_encoder_attention_heads
        self.num_heads = args.encoder_attention_heads
        self.select_strategy = args.attn_head_select_strategy

        super().__init__(args, dictionary, embed_tokens)
        self.attn_head_selector = AttnHeadSelector(
            self.num_tasks,
            self.num_layers,
            self.total_num_heads,
            self.num_heads,
            self.select_strategy
        )
        self.task_ids = None
        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args, i) for i in range(args.encoder_layers)]
        )

    def set_task_ids(self, task_ids):
        self.task_ids = task_ids

    def build_encoder_layer(self, args, layer_idx=None):
        return HeadSelectionTransformerEncoderLayer(
            args,
            layer_idx,
            attn_head_selector=self.attn_head_selector
        )

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        self.attn_head_selector.head_select(self.task_ids)
        return super().forward(src_tokens, src_lengths, return_all_hiddens, token_embeddings)


class HeadSelectionTransformerDecoder(TransformerDecoder):

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.num_tasks = args.decoder_tasks
        self.num_layers = args.decoder_layers
        self.total_num_heads = args.total_decoder_attention_heads
        self.num_heads = args.decoder_attention_heads
        self.select_strategy = args.attn_head_select_strategy
        super().__init__(
            args, dictionary, embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection
        )
        self.self_attn_head_selector = None
        self.enc_attn_head_selector = None
        if safe_hasattr(args, "decoder_self_attn_head_select") and args.decoder_self_attn_head_select:
            self.self_attn_head_selector = AttnHeadSelector(
                self.num_tasks,
                self.num_layers,
                self.total_num_heads,
                self.num_heads,
                self.select_strategy
            )
        if safe_hasattr(args, "dec_enc_attn_head_select") and args.dec_enc_attn_head_select:
            self.enc_attn_head_selector = AttnHeadSelector(
                self.num_tasks,
                self.num_layers,
                self.total_num_heads,
                self.num_heads,
                self.select_strategy
            )
        self.task_ids = None
        self.layers = nn.ModuleList(
            [
                self.build_head_selection_decoder_layer(args, no_encoder_attn, idx) for idx in range(args.decoder_layers)
            ]
        )

    def set_task_ids(self, task_ids):
        self.task_ids = task_ids

    def build_head_selection_decoder_layer(self, args, no_encoder_attn=False, layer_idx=None):
        return HeadSelectionTransformerDecoderLayer(
            args,
            layer_idx,
            self.self_attn_head_selector,
            self.enc_attn_head_selector,
            no_encoder_attn=no_encoder_attn
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        if self.self_attn_head_selector is not None:
            self.self_attn_head_selector.head_select(self.task_ids)
        if self.enc_attn_head_selector is not None:
            self.enc_attn_head_selector.head_select(self.task_ids)
        return super().forward(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            features_only=features_only,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
