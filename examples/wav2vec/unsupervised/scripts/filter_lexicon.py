#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from fairseq.data import Dictionary


def get_parser():
    parser = argparse.ArgumentParser(
        description="filters a lexicon given a unit dictionary"
    )
    parser.add_argument("-d", "--unit-dict", help="unit dictionary", required=True)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    d = Dictionary.load(args.unit_dict)
    symbols = set(d.symbols)

    for line in sys.stdin:
        items = line.rstrip().split()
        skip = len(items) < 2
        for x in items[1:]:
            if x not in symbols:
                skip = True
                break
        if not skip:
            print(line, end="")


if __name__ == "__main__":
    main()
