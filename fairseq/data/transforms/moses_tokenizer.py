# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.data.transforms import register_tokenizer


@register_tokenizer('moses')
class MosesTokenizer(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('-s', '--source-lang', default='en', metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default='en', metavar='TARGET',
                            help='target language')
        parser.add_argument('--aggressive-dash-splits', action='store_true', default=False,
                            help='triggers dash split rules')
        parser.add_argument('--no-escape', action='store_true', default=False,
                            help='don\'t perform HTML escaping on apostrophy, quotes, etc.')
        # fmt: on

    def __init__(self, args):
        self.args = args
        try:
            from sacremoses import MosesTokenizer, MosesDetokenizer
            self.tok = MosesTokenizer(args.source_lang)
            self.detok = MosesDetokenizer(args.target_lang)
        except ImportError:
            raise ImportError('Please install Moses tokenizer with: pip install sacremoses')

    def encode(self, x: str) -> str:
        return self.tok.tokenize(
            x,
            aggressive_dash_splits=self.args.aggressive_dash_splits,
            return_str=True,
            escape=(not self.args.no_escape),
        )

    def decode(self, x: str) -> str:
        return self.detok.detokenize(x.split())
