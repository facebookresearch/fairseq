#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Use: echo {text} | python tokenize_indic.py {language}

import sys

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize import trivial_tokenize


factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer(
    sys.argv[1], remove_nuktas=False, nasals_mode="do_nothing"
)

for line in sys.stdin:
    normalized_line = normalizer.normalize(line.strip())
    tokenized_line = " ".join(trivial_tokenize(normalized_line, sys.argv[1]))
    print(tokenized_line)
