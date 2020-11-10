#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        root = next(tsv).strip()
        for line in tsv:
            line = line.strip()
            dir = os.path.dirname(line)
            if dir not in transcriptions:
                parts = dir.split(os.path.sep)
                trans_path = f"{parts[-2]}-{parts[-1]}.trans.txt"
                path = os.path.join(root, dir, trans_path)
                assert os.path.exists(path)
                texts = {}
                with open(path, "r") as trans_f:
                    for tline in trans_f:
                        items = tline.strip().split()
                        texts[items[0]] = " ".join(items[1:])
                transcriptions[dir] = texts
            part = os.path.basename(line).split(".")[0]
            assert part in transcriptions[dir]
            print(transcriptions[dir][part], file=wrd_out)
            print(
                " ".join(list(transcriptions[dir][part].replace(" ", "|"))) + " |",
                file=ltr_out,
            )


if __name__ == "__main__":
    main()
