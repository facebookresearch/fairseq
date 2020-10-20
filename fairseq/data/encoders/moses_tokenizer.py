# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@dataclass
class MosesTokenizerConfig(FairseqDataclass):
    source_lang: str = field(default="en", metadata={"help": "source language"})
    target_lang: str = field(default="en", metadata={"help": "target language"})
    moses_no_dash_splits: bool = field(
        default=False, metadata={"help": "don't apply dash split rules"}
    )
    moses_no_escape: bool = field(
        default=False,
        metadata={"help": "don't perform HTML escaping on apostrophe, quotes, etc."},
    )


@register_tokenizer("moses", dataclass=MosesTokenizerConfig)
class MosesTokenizer(object):
    def __init__(self, cfg):
        self.cfg = cfg

        try:
            from sacremoses import MosesTokenizer, MosesDetokenizer

            self.tok = MosesTokenizer(cfg.source_lang)
            self.detok = MosesDetokenizer(cfg.target_lang)
        except ImportError:
            raise ImportError(
                "Please install Moses tokenizer with: pip install sacremoses"
            )

    def encode(self, x: str) -> str:
        return self.tok.tokenize(
            x,
            aggressive_dash_splits=(not self.cfg.moses_no_dash_splits),
            return_str=True,
            escape=(not self.cfg.moses_no_escape),
        )

    def decode(self, x: str) -> str:
        return self.detok.detokenize(x.split())
