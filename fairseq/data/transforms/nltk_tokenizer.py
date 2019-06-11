# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.data.transforms import register_tokenizer


@register_tokenizer('nltk')
class NLTKTokenizer(object):

    def __init__(self, source_lang=None, target_lang=None):
        try:
            from nltk.tokenize import word_tokenize
            self.word_tokenize = word_tokenize
        except ImportError:
            raise ImportError('Please install nltk with: pip install nltk')

    def encode(self, x: str) -> str:
        return ' '.join(self.word_tokenize(x))

    def decode(self, x: str) -> str:
        return x
