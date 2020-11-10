# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Linformer: Self-Attention with Linear Complexity
"""

import logging

from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import RobertaEncoder, RobertaModel

from ..modules.linformer_sentence_encoder import LinformerSentenceEncoder


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

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = LinformerEncoder(args, task.source_dictionary)
        return cls(args, encoder)


class LinformerEncoder(RobertaEncoder):
    """Linformer encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

        self.sentence_encoder = LinformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            compressed=args.compressed,
            shared_kv_compressed=args.shared_kv_compressed,
            shared_layer_kv_compressed=args.shared_layer_kv_compressed,
            freeze_compress=args.freeze_compress,
        )


@register_model_architecture("linformer_roberta", "linformer_roberta")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.compressed = getattr(args, "compressed", 4)
    args.shared_kv_compressed = getattr(args, "shared_kv_compressed", 0)
    args.shared_layer_kv_compressed = getattr(args, "shared_layer_kv_compressed", 0)
    args.freeze_compress = getattr(args, "freeze_compress", 0)


@register_model_architecture("linformer_roberta", "linformer_roberta_base")
def linformer_roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture("linformer_roberta", "linformer_roberta_large")
def linformer_roberta_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.compressed = getattr(args, "compressed", 4)
    args.shared_kv_compressed = getattr(args, "shared_kv_compressed", 0)
    args.shared_layer_kv_compressed = getattr(args, "shared_layer_kv_compressed", 0)
