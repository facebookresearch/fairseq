# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data.encoders import register_tokenizer


@register_tokenizer('moses')
class MosesTokenizer(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--moses-source-lang', default='en', metavar='SRC',
                            help='source language')
        parser.add_argument('--moses-target-lang', default='en', metavar='TARGET',
                            help='target language')
        parser.add_argument('--moses-no-dash-splits', action='store_true', default=False,
                            help='don\'t apply dash split rules')
        parser.add_argument('--moses-no-escape', action='store_true', default=False,
                            help='don\'t perform HTML escaping on apostrophy, quotes, etc.')
        # fmt: on

    def __init__(self, args):
        self.args = args
        try:
            from sacremoses import MosesTokenizer, MosesDetokenizer
            self.tok = MosesTokenizer(args.moses_source_lang)
            self.detok = MosesDetokenizer(args.moses_target_lang)
        except ImportError:
            raise ImportError('Please install Moses tokenizer with: pip install sacremoses')

    def encode(self, x: str) -> str:
        return self.tok.tokenize(
            x,
            aggressive_dash_splits=(not self.args.moses_no_dash_splits),
            return_str=True,
            escape=(not self.args.moses_no_escape),
        )

    def decode(self, x: str) -> str:
        return self.detok.detokenize(x.split())
