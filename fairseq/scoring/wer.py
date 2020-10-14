# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.scoring import register_scorer, BaseScorer
from fairseq.scoring.tokenizer import EvaluationTokenizer


@register_scorer("wer")
class WerScorer(BaseScorer):
    def __init__(self, args):
        super().__init__(args)
        self.reset()
        try:
            import editdistance as ed
        except ImportError:
            raise ImportError('Please install editdistance to use WER scorer')
        self.ed = ed
        self.tokenizer = EvaluationTokenizer(
            tokenizer_type=self.args.wer_tokenizer,
            lowercase=self.args.wer_lowercase,
            punctuation_removal=self.args.wer_remove_punct,
            character_tokenization=self.args.wer_char_level,
        )

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--wer-tokenizer', type=str, default='none',
                            choices=EvaluationTokenizer.ALL_TOKENIZER_TYPES,
                            help='sacreBLEU tokenizer to use for evaluation')
        parser.add_argument('--wer-remove-punct', action='store_true',
                            help='remove punctuation')
        parser.add_argument('--wer-char-level', action='store_true',
                            help='evaluate at character level')
        parser.add_argument('--wer-lowercase', action='store_true',
                            help='lowercasing')
        # fmt: on

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
        return (
            100.0 * self.distance / self.ref_length if self.ref_length > 0
            else 0
        )
