# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models.transformer.transformer_legacy import TransformerModel
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
)

from .transformer_decoder import PointerGeneratorTransformerDecoder
from .transformer_encoder import PointerGeneratorTransformerEncoder


class PointerGeneratorTransformerModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return PointerGeneratorTransformerEncoder(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return PointerGeneratorTransformerDecoder(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )
