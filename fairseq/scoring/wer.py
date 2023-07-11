# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer
from fairseq.scoring.tokenizer import EvaluationTokenizer


@dataclass
class WerScorerConfig(FairseqDataclass):
    wer_tokenizer: EvaluationTokenizer.ALL_TOKENIZER_TYPES = field(
        default="none", metadata={"help": "sacreBLEU tokenizer to use for evaluation"}
    )
    wer_remove_punct: bool = field(
        default=False, metadata={"help": "remove punctuation"}
    )
    wer_char_level: bool = field(
        default=False, metadata={"help": "evaluate at character level"}
    )
    wer_lowercase: bool = field(default=False, metadata={"help": "lowercasing"})


@register_scorer("wer", dataclass=WerScorerConfig)
class WerScorer(BaseScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.reset()
        try:
            import editdistance as ed
        except ImportError:
            raise ImportError("Please install editdistance to use WER scorer")
        self.ed = ed
        self.tokenizer = EvaluationTokenizer(
            tokenizer_type=self.cfg.wer_tokenizer,
            lowercase=self.cfg.wer_lowercase,
            punctuation_removal=self.cfg.wer_remove_punct,
            character_tokenization=self.cfg.wer_char_level,
        )

    def reset(self):
        self.distance = 0
        self.ref_length = 0

    def add_string(self, ref, pred):
        ref_items = self.tokenizer.tokenize(ref).split()
        pred_items = self.tokenizer.tokenize(pred).split()
        self.distance += self.ed.eval(ref_items, pred_items)
        self.ref_length += len(ref_items)

    def result_string(self):
        return f"WER: {self.score():.2f}"

    def score(self):
        return 100.0 * self.distance / self.ref_length if self.ref_length > 0 else 0
