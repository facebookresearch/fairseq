# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@dataclass
class MarianTokenizerConfig(FairseqDataclass):
    tokenizer_path: str = field(default="/opt/ml/input/data/model", metadata={"help": "source of tokenizer files"})
    

@register_tokenizer("marian", dataclass=MarianTokenizerConfig)
class MarianTokenizer(object):
    def __init__(self, cfg):
        try:
            from transformers import MarianTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers with: pip install transformers"
            )
        self.marian_tokenizer = MarianTokenizer.from_pretrained(cfg.tokenizer_path)

    def encode(self, x: str) -> str:
        return " ".join(self.marian_tokenizer.tokenize(x))


    def decode(self, x: str) -> str:
        return self.marian_tokenizer.clean_up_tokenization(
            self.marian_tokenizer.convert_tokens_to_string(x.split(" "))
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith("##")
