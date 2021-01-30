#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="sentencepiece model to use for decoding"
    )
    parser.add_argument("--input", required=True, help="input file to decode")
    parser.add_argument("--input_format", choices=["piece", "id"], default="piece")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    if args.input_format == "piece":

        def decode(l):
            return "".join(sp.DecodePieces(l))

    elif args.input_format == "id":

        def decode(l):
            return "".join(sp.DecodeIds(l))

    else:
        raise NotImplementedError

    def tok2int(tok):
        # remap reference-side <unk> (represented as <<unk>>) to 0
        return int(tok) if tok != "<<unk>>" else 0

    with open(args.input, "r", encoding="utf-8") as h:
        for line in h:
            if args.input_format == "id":
                print(decode(list(map(tok2int, line.rstrip().split()))))
            elif args.input_format == "piece":
                print(decode(line.rstrip().split()))


if __name__ == "__main__":
    main()
