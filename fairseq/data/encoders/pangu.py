from dataclasses import dataclass, field
from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass
from typing import List
import sys, os, re
from pangu import spacing

RE_WS_IN_FW = re.compile(r'([\u2018\u2019\u201c\u201d\u2e80-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef])\s+(?=[\u2018\u2019\u201c\u201d\u2e80-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef])')


@dataclass
class PanguTokenizerConfig(FairseqDataclass):
    target_lang: str = field(default="zh", metadata={"help": "target language"})


@register_tokenizer("pangu", dataclass=PanguTokenizerConfig)
class PanguTokenizer(object):
    def __init__(self, cfg: PanguTokenizerConfig):
        self.cfg = cfg
        
    def encode(self, text: str) -> str:
        return text

    def decode(self, text: str) -> str:
        
        text = spacing(RE_WS_IN_FW.sub(r'\1', text)).strip()
        return text
