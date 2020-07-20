# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

from fairseq.model_parallel.modules import ModelParallelMultiheadAttention

try:
    from fairseq.model_parallel.megatron.mpu import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


class ModelParallelTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block over multiple gpus.

        See "Megatron-LM: https://arxiv.org/pdf/1909.08053.pdf" for more details.
    """

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if q_noise > 0:
            raise NotImplementedError
        return ColumnParallelLinear(input_dim, output_dim, gather_output=False)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if q_noise > 0:
            raise NotImplementedError
        return RowParallelLinear(input_dim, output_dim, input_is_parallel=True)

    def build_self_attention(self, embed_dim, args, **unused_kwargs):
        return ModelParallelMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )


class ModelParallelTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer block.

        See "Megatron-LM: https://arxiv.org/pdf/1909.08053.pdf" for more details.
    """
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if q_noise > 0:
            raise NotImplementedError
        return ColumnParallelLinear(input_dim, output_dim, gather_output=False)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if q_noise > 0:
            raise NotImplementedError
        return RowParallelLinear(input_dim, output_dim, input_is_parallel=True)

    def build_self_attention(self, embed_dim, args, **unused_kwargs):
        return ModelParallelMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=not getattr(args, "cross_self_attention", False),
        )

    def build_encoder_attention(self, embed_dim, args, **unused_kwargs):
        return ModelParallelMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )
