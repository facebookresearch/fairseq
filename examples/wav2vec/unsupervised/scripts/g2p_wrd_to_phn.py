#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from g2p_en import G2p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compact",
        action="store_true",
        help="if set, compacts phones",
    )
    args = parser.parse_args()

    compact = args.compact

    wrd_to_phn = {}
    g2p = G2p()
    for line in sys.stdin:
        words = line.strip().split()
        phones = []
        for w in words:
            if w not in wrd_to_phn:
                wrd_to_phn[w] = g2p(w)
                if compact:
                    wrd_to_phn[w] = [
                        p[:-1] if p[-1].isnumeric() else p for p in wrd_to_phn[w]
                    ]
            phones.extend(wrd_to_phn[w])
        try:
            print(" ".join(phones))
        except:
            print(wrd_to_phn, words, phones, file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
