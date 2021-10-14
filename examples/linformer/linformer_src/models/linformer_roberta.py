# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Linformer: Self-Attention with Linear Complexity
"""

import logging

import torch
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import (
    init_bert_params,
    roberta_base_architecture,
    roberta_large_architecture,
    RobertaEncoder,
    RobertaModel,
)
from fairseq.utils import safe_hasattr

from ..modules.linformer_sentence_encoder import LinformerTransformerEncoder


logger = logging.getLogger(__name__)


@register_model("linformer_roberta")
class LinformerModel(RobertaModel):
    @staticmethod
    def add_args(parser):
        RobertaModel.add_args(parser)

        # add args for Linformer
        parser.add_argument(
            "--compressed", type=int, help="compressed ratio of sequence length"
        )
        parser.add_argument(
            "--shared-kv-compressed",
            type=int,
            help="share compressed matrix between k and v, in each layer",
        )
        parser.add_argument(
            "--shared-layer-kv-compressed",
            type=int,
            help="share compressed matrix between k and v and across all layers",
        )
        parser.add_argument(
            "--freeze-compress",
            type=int,
            help="freeze the parameters in compressed layer",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = LinformerEncoder(args, task.source_dictionary)
        return cls(args, encoder)


class LinformerEncoder(RobertaEncoder):
    """Linformer encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.register_buffer("version", torch.tensor(2))

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = LinformerTransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        prefix = name + "." if name != "" else ""

        # some old checkpoints had weight sharing implemented incorrectly
        # (note: this was correct in the original paper code)
        if utils.item(state_dict.get(f"{prefix}version", torch.tensor(1))) < 2:
            state_dict[f"{prefix}version"] = torch.tensor(1)
            # check if input embeddings and output embeddings were tied
            if not torch.allclose(
                state_dict[f"{prefix}sentence_encoder.embed_tokens.weight"],
                state_dict[f"{prefix}lm_head.weight"],
            ):
                # they weren't tied, re-init the LM head without weight sharing
                self.lm_head = self.build_lm_head(
                    embed_dim=self.args.encoder_embed_dim,
                    output_dim=len(self.dictionary),
                    activation_fn=self.args.activation_fn,
                    weight=None,  # don't share weights
                )


@register_model_architecture("linformer_roberta", "linformer_roberta")
def base_architecture(args):
    args.compressed = getattr(args, "compressed", 4)
    args.shared_kv_compressed = getattr(args, "shared_kv_compressed", 0)
    args.shared_layer_kv_compressed = getattr(args, "shared_layer_kv_compressed", 0)
    args.freeze_compress = getattr(args, "freeze_compress", 0)
    roberta_base_architecture(args)


@register_model_architecture("linformer_roberta", "linformer_roberta_base")
def linformer_roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture("linformer_roberta", "linformer_roberta_large")
def linformer_roberta_large_architecture(args):
    roberta_large_architecture(args)
    base_architecture(args)
