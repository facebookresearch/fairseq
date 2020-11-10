# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


@dataclass
class HuggingFaceByteLevelBPEConfig(FairseqDataclass):
    bpe_merges: str = field(default="???", metadata={"help": "path to merges.txt"})
    bpe_vocab: str = field(default="???", metadata={"help": "path to vocab.json"})
    bpe_add_prefix_space: bool = field(
        default=False, metadata={"help": "add prefix space before encoding"}
    )


@register_bpe("hf_byte_bpe", dataclass=HuggingFaceByteLevelBPEConfig)
class HuggingFaceByteLevelBPE(object):
    def __init__(self, cfg):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                "Please install huggingface/tokenizers with: " "pip install tokenizers"
            )

        self.bpe = ByteLevelBPETokenizer(
            cfg.bpe_vocab,
            cfg.bpe_merges,
            add_prefix_space=cfg.bpe_add_prefix_space,
        )

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x).ids))

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")
