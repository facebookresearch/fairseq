from dataclasses import dataclass, field
from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass
from typing import List
import json

@dataclass
class MecabTokenizerConfig(FairseqDataclass):
    target_lang: str = field(default="ko", metadata={"help": "target language"})


@register_tokenizer("mecab", dataclass=MecabTokenizerConfig)
class MecabTokenizer(object):
    def __init__(self, cfg: MecabTokenizerConfig):
        self.cfg = cfg

        if self.cfg.target_lang == 'ko': 
            try:
                import MeCab
                import mecab_ko_dic

                self.mecab = MeCab.Tagger(mecab_ko_dic.MECAB_ARGS)
                self.config = {"space_symbol": "▃"}
            except ImportError:
                raise ImportError(
                    "Please install Mecab Ko dictionary"
                )


    def encode(self, text: str) -> str:
        text = text.strip()
        text_ptr = 0
        tokenized = []
        for mor in self.mecab.parse(text).split("\n"):
            if "\t" in mor:
                splitted = mor.split("\t")
                token = splitted[0]
                # pos = splitted[1].split(",", 1)[0]

                if text[text_ptr] == " ":
                    while text[text_ptr] == " ":
                        text_ptr += 1
                    assert (
                        text[text_ptr] == token[0]
                    ), f"{repr(text)}//{text_ptr}//{text[text_ptr]}//{token}//{token[0]}\n"

                    tokenized.append(self.config["space_symbol"])

                tokenized.append(token)
                text_ptr += len(token)
            return " ".join(tokenized)

    def decode(self, text: str) -> str:
        text = "".join(text.split(" ")).replace("▃", " ").strip()
        return text
