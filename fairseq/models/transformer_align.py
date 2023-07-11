# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big,
)


@register_model("transformer_align")
class TransformerAlignModel(TransformerModel):
    """
    See "Jointly Learning to Align and Translate with Transformer
    Models" (Garg et al., EMNLP 2019).
    """

    def __init__(self, encoder, decoder, args):
        super().__init__(args, encoder, decoder)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        self.full_context_alignment = args.full_context_alignment

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(TransformerAlignModel, TransformerAlignModel).add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='D',
                            help='Number of cross attention heads per layer to supervised with alignments')
        parser.add_argument('--alignment-layer', type=int, metavar='D',
                            help='Layer number which has to be supervised. 0 corresponding to the bottommost layer.')
        parser.add_argument('--full-context-alignment', action='store_true',
                            help='Whether or not alignment is supervised conditioned on the full target context.')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        transformer_align(args)

        transformer_model = TransformerModel.build_model(args, task)
        return TransformerAlignModel(
            transformer_model.encoder, transformer_model.decoder, args
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        return self.forward_decoder(prev_output_tokens, encoder_out)

    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        attn_args = {
            "alignment_layer": self.alignment_layer,
            "alignment_heads": self.alignment_heads,
        }
        decoder_out = self.decoder(prev_output_tokens, encoder_out, **attn_args)

        if self.full_context_alignment:
            attn_args["full_context_alignment"] = self.full_context_alignment
            _, alignment_out = self.decoder(
                prev_output_tokens,
                encoder_out,
                features_only=True,
                **attn_args,
                **extra_args,
            )
            decoder_out[1]["attn"] = alignment_out["attn"]

        return decoder_out


@register_model_architecture("transformer_align", "transformer_align")
def transformer_align(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    args.full_context_alignment = getattr(args, "full_context_alignment", False)
    base_architecture(args)


@register_model_architecture("transformer_align", "transformer_wmt_en_de_big_align")
def transformer_wmt_en_de_big_align(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    transformer_wmt_en_de_big(args)
