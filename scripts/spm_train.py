#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import sentencepiece as spm


if __name__ == "__main__":
    spm.SentencePieceTrainer.Train(" ".join(sys.argv[1:]))
