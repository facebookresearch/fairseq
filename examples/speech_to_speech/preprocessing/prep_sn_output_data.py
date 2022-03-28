#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from tqdm import tqdm


def process(args):
    args.output_root.mkdir(exist_ok=True)

    # load units
    units = {}
    with open(args.in_unit) as f:
        for line in f:
            unit_seq, utt_id = line.strip().rsplit(" ", 1)
            utt_id = int(utt_id[6:-1])  # remove "(None-"
            units[utt_id] = unit_seq

    with open(args.in_audio) as f, open(
        args.output_root / f"{args.in_audio.stem}.txt", "w"
    ) as o:
        f.readline()
        for i, line in enumerate(tqdm(f.readlines())):
            audio, _ = line.strip().split("\t", 1)
            sample_id = Path(audio).stem
            o.write(f"{sample_id}|{units[i]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-unit",
        required=True,
        type=Path,
        help="unit file (output from the speech normalizer)",
    )
    parser.add_argument(
        "--in-audio",
        required=True,
        type=Path,
        help="tsv file (input to the normalizer)",
    )
    parser.add_argument(
        "--output-root", required=True, type=Path, help="output directory"
    )

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
