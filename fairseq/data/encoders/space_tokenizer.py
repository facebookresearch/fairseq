# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import re

from fairseq.data.encoders import register_tokenizer


@register_tokenizer('space')
class SpaceTokenizer(object):

    def __init__(self, source_lang=None, target_lang=None):
        self.space_tok = re.compile(r"\s+")

    def encode(self, x: str) -> str:
        return self.space_tok.sub(' ', x)

    def decode(self, x: str) -> str:
        return x
