#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Evaluate the perplexity of a trained language model.
"""
import sys

from fairseq.cli import eval_lm_main

if __name__ == '__main__':
    print("WARNING - The eval_lm.py script has been deprecated. This is going to be removed in future release. "
          "Please use the fairseq-lmeval script instead.", file=sys.stderr, flush=True)
    eval_lm_main()
