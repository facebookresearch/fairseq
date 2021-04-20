#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

from sacremoses.normalize import MosesPunctNormalizer


def main(args):
    normalizer = MosesPunctNormalizer(lang=args.lang, penn=args.penn)
    for line in sys.stdin:
        print(normalizer.normalize(line.rstrip()), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", "-l", default="en")
    parser.add_argument("--penn", "-p", action="store_true")
    args = parser.parse_args()

    main(args)
