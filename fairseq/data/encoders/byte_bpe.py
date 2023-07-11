# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.data.encoders.byte_utils import (
    SPACE,
    SPACE_ESCAPE,
    byte_encode,
    smart_byte_decode,
)
from fairseq.dataclass import FairseqDataclass


@dataclass
class ByteBpeConfig(FairseqDataclass):
    sentencepiece_model_path: str = field(
        default="???", metadata={"help": "path to sentencepiece model"}
    )


@register_bpe("byte_bpe", dataclass=ByteBpeConfig)
class ByteBPE(object):
    def __init__(self, cfg):
        vocab = file_utils.cached_path(cfg.sentencepiece_model_path)
        try:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(vocab)
        except ImportError:
            raise ImportError(
                "Please install sentencepiece with: pip install sentencepiece"
            )

    def encode(self, x: str) -> str:
        byte_encoded = byte_encode(x)
        return SPACE.join(self.sp.EncodeAsPieces(byte_encoded))

    @staticmethod
    def decode(x: str) -> str:
        unescaped = x.replace(SPACE, "").replace(SPACE_ESCAPE, SPACE)
        return smart_byte_decode(unescaped)
