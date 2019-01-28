#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""
import sys

from fairseq.cli import train_main

if __name__ == '__main__':
    print("WARNING - The train.py script has been deprecated. This is going to be removed in future release. "
          "Please use the fairseq-train script instead.", file=sys.stderr, flush=True)
    train_main()
