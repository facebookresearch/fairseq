#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fileinput

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract back-translations from the stdout of fairseq-generate. "
            "If there are multiply hypotheses for a source, we only keep the first one. "
        )
    )
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument(
        "--srclang", required=True, help="source language (extracted from H-* lines)"
    )
    parser.add_argument(
        "--tgtlang", required=True, help="target language (extracted from S-* lines)"
    )
    parser.add_argument("--minlen", type=int, help="min length filter")
    parser.add_argument("--maxlen", type=int, help="max length filter")
    parser.add_argument("--ratio", type=float, help="ratio filter")
    parser.add_argument("files", nargs="*", help="input files")
    args = parser.parse_args()

    def validate(src, tgt):
        srclen = len(src.split(" ")) if src != "" else 0
        tgtlen = len(tgt.split(" ")) if tgt != "" else 0
        if (
            (args.minlen is not None and (srclen < args.minlen or tgtlen < args.minlen))
            or (
                args.maxlen is not None
                and (srclen > args.maxlen or tgtlen > args.maxlen)
            )
            or (
                args.ratio is not None
                and (max(srclen, tgtlen) / float(min(srclen, tgtlen)) > args.ratio)
            )
        ):
            return False
        return True

    def safe_index(toks, index, default):
        try:
            return toks[index]
        except IndexError:
            return default

    with open(args.output + "." + args.srclang, "w") as src_h, open(
        args.output + "." + args.tgtlang, "w"
    ) as tgt_h:
        for line in tqdm(fileinput.input(args.files)):
            if line.startswith("S-"):
                tgt = safe_index(line.rstrip().split("\t"), 1, "")
            elif line.startswith("H-"):
                if tgt is not None:
                    src = safe_index(line.rstrip().split("\t"), 2, "")
                    if validate(src, tgt):
                        print(src, file=src_h)
                        print(tgt, file=tgt_h)
                    tgt = None


if __name__ == "__main__":
    main()
