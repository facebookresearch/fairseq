#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

from pythainlp import word_tokenize


for line in sys.stdin:
    print(" ".join(word_tokenize(line.strip())))
