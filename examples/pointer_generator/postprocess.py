#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import re
import argparse


class OOVIndexError(IndexError):
    def __init__(self, pos, source_seq, target_seq):
        super(OOVIndexError, self).__init__(
            "A <unk-N> tag in the target sequence refers to a position that is "
            "outside the source sequence. Most likely there was a mismatch in "
            "provided source and target sequences. Otherwise this would mean that "
            "the pointing mechanism somehow attended to a position that is past "
            "the actual sequence end."
        )
        self.source_pos = pos
        self.source_seq = source_seq
        self.target_seq = target_seq


def replace_oovs(source_in, target_in, target_out):
    """Replaces <unk-N> tokens in the target text with the corresponding word in
  the source text.
  """

    oov_re = re.compile("^<unk-([0-9]+)>$")

    for source_seq, target_seq in zip(source_in, target_in):
        target_seq_out = []

        pos_to_word = source_seq.strip().split()
        for token in target_seq.strip().split():
            m = oov_re.match(token)
            if m:
                pos = int(m.group(1))
                if pos >= len(pos_to_word):
                    raise OOVIndexError(pos, source_seq, target_seq)
                token_out = pos_to_word[pos]
            else:
                token_out = token
            target_seq_out.append(token_out)
        target_out.write(" ".join(target_seq_out) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replaces <unk-N> tokens in target sequences with words from "
        "the corresponding position in the source sequence."
    )
    parser.add_argument(
        "--source", type=str, help="text file with source sequences", required=True
    )
    parser.add_argument(
        "--target", type=str, help="text file with target sequences", required=True
    )
    parser.add_argument(
        "--target-out",
        type=str,
        help="where to write target sequences without <unk-N> " "entries",
        required=True,
    )
    args = parser.parse_args()

    target_in = (
        open(args.target, "r", encoding="utf-8") if args.target is not None else None
    )
    target_out = (
        open(args.target_out, "w", encoding="utf-8")
        if args.target_out is not None
        else None
    )
    with open(args.source, "r", encoding="utf-8") as source_in, open(
        args.target, "r", encoding="utf-8"
    ) as target_in, open(args.target_out, "w", encoding="utf-8") as target_out:
        replace_oovs(source_in, target_in, target_out)


if __name__ == "__main__":
    try:
        main()
    except OOVIndexError as e:
        print(e, file=sys.stderr)
        print("Source sequence:", e.source_seq.strip(), file=sys.stderr)
        print("Target sequence:", e.target_seq.strip(), file=sys.stderr)
        print(
            "Source sequence length:",
            len(e.source_seq.strip().split()),
            file=sys.stderr,
        )
        print("The offending tag points to:", e.source_pos)
        sys.exit(2)
