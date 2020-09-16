#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unicodedata
import math

from fairseq.scorers import BaseScorer, register_scorer


class EvalTokenizer(object):
    def __init__(self, punctuation_removal: bool = False):
        self.punctuation_removal = punctuation_removal

    @classmethod
    def remove_punctuation(cls, sent: str, remove_all: bool = False):
        if remove_all:
            return ''.join(c for c in sent if unicodedata.category(c)[0] != 'P')
        return ' '.join(
            t for t in sent.split(' ')
            if not all(unicodedata.category(c)[0] == 'P' for c in t)
        )

    def tokenize(self, sent: str, lang: str = 'en'):
        from sacrebleu.tokenizers import TOKENIZERS

        if lang == 'zh':
            tokenizer = TOKENIZERS['zh']
        elif lang == 'ja':
            tokenizer = TOKENIZERS['ja-mecab']
        else:
            tokenizer = TOKENIZERS['13a']

        tokenized = tokenizer()(sent)

        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)

        return tokenized


@register_scorer('fastwer')
class FastWERScorer(BaseScorer):
    def __init__(self, args):
        super(FastWERScorer, self).__init__(args)
        self.lang = args.fastwer_lang
        self.eval_tokenizer = EvalTokenizer(punctuation_removal=True)

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--fastwer-lang', type=str, default='en',
                            help='Language of evaluated text '
                                 '(for auto-selection of fastwer tokenizer)')
        # fmt: on

    def preprocess(self, sent: str):
        tokenized = self.eval_tokenizer.tokenize(sent, self.lang)
        punct_removed = self.eval_tokenizer.remove_punctuation(tokenized)
        return punct_removed.lower()

    def add_string(self, ref: str, pred: str):
        self.ref.append(self.preprocess(ref))
        self.pred.append(self.preprocess(pred))

    def score(self, char_level: bool = False) -> float:
        try:
            import fastwer as fw
        except ImportError:
            raise ImportError('Please install fastwer to use FastWER scorer')
        return fw.score(self.pred, self.ref, char_level=char_level)

    def bp_result_string(self):
        ref_len, hypo_len = 0, 0
        for r, p in zip(self.ref, self.pred):
            ref_len += len(r.split(' '))
            hypo_len += len(p.split(' '))
        r = ref_len / hypo_len
        bp = min(1, math.exp(1 - r))
        return f' (BP={bp:.3f}, ratio={r:.3f}, syslen={hypo_len}, reflen={ref_len})'

    def result_string(self, char_level: bool = False) -> str:
        metric_name = 'CER' if char_level else 'WER'
        corpus_score = self.score(char_level=char_level)
        return f'{metric_name} = {corpus_score:.2f}' + self.bp_result_string()
