#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import sacremoses


def main(args):
    """Tokenizes, preserving tabs"""
    mt = sacremoses.MosesTokenizer(lang=args.lang)

    def tok(s):
        return mt.tokenize(s, return_str=True)

    for line in sys.stdin:
        parts = list(map(tok, line.split("\t")))
        print(*parts, sep="\t", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", "-l", default="en")
    parser.add_argument("--penn", "-p", action="store_true")
    parser.add_argument("--fields", "-f", help="fields to tokenize")
    args = parser.parse_args()

    main(args)
