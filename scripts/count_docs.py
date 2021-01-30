#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Count the number of documents and average number of lines and tokens per
document in a large file. Documents should be separated by a single empty line.
"""

import argparse
import gzip
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--gzip", action="store_true")
    args = parser.parse_args()

    def gopen():
        if args.gzip:
            return gzip.open(args.input, "r")
        else:
            return open(args.input, "r", encoding="utf-8")

    num_lines = []
    num_toks = []
    with gopen() as h:
        num_docs = 1
        num_lines_in_doc = 0
        num_toks_in_doc = 0
        for i, line in enumerate(h):
            if len(line.strip()) == 0:  # empty line indicates new document
                num_docs += 1
                num_lines.append(num_lines_in_doc)
                num_toks.append(num_toks_in_doc)
                num_lines_in_doc = 0
                num_toks_in_doc = 0
            else:
                num_lines_in_doc += 1
                num_toks_in_doc += len(line.rstrip().split())
            if i % 1000000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100000 == 0:
                print(".", file=sys.stderr, end="", flush=True)
        print(file=sys.stderr, flush=True)

    print("found {} docs".format(num_docs))
    print("average num lines per doc: {}".format(np.mean(num_lines)))
    print("average num toks per doc: {}".format(np.mean(num_toks)))


if __name__ == "__main__":
    main()
