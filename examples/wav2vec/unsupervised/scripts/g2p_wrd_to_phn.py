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
    parser.add_argument("root_dirs", nargs="*")
    parser.add_argument("--insert-silence", "-s", action="store_true")
    args = parser.parse_args()
    sil = "<s>"

    wrd_to_phn = {}
    g2p = G2p()
    for line in sys.stdin:
        words = line.strip().split()
        phones = []
        if args.insert_silence:
            phones.append(sil)
        for w in words:
            if w not in wrd_to_phn:
                wrd_to_phn[w] = g2p(w)
            phones.extend(wrd_to_phn[w])
            if args.insert_silence:
                phones.append(sil)
        try:
            print(" ".join(phones))
        except:
            print(wrd_to_phn, w, phones, file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
