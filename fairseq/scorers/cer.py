#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.scorers import register_scorer
from fairseq.scorers.fastwer import FastWERScorer


@register_scorer('fastcer')
class FastCERScorer(FastWERScorer):
    def score(self, char_level: bool = False) -> float:
        return super(FastCERScorer, self).score(char_level=True)

    def result_string(self) -> str:
        return super(FastCERScorer, self).result_string(char_level=True)
