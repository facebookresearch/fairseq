#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Extracts random constraints from reference files."""

import argparse
import random
import sys
from sacrebleu import extract_ngrams


def get_phrase(words, index, length):
    assert(index < len(words) - length + 1)
    phr = ' '.join(words[index:index+length])
    for i in range(index, index + length):
        words.pop(index)
    return phr


def main(args):

    if args.seed:
        random.seed(args.seed)

    for line in sys.stdin:
        constraints = []

        def add_constraint(constraint):
            constraints.append(constraint)

        source = line.rstrip()
        if '\t' in line:
            source, target = line.split('\t')
            if args.add_sos:
                target = f"<s> {target}"
            if args.add_eos:
                target = f"{target} </s>"

            if len(target.split()) >= args.len:
                words = [target]

                num = args.number

                choices = {}
                for i in range(num):
                    if len(words) == 0:
                        break
                    segmentno = random.choice(range(len(words)))
                    segment = words.pop(segmentno)
                    tokens = segment.split()
                    phrase_index = random.choice(range(len(tokens)))
                    choice = " ".join(tokens[phrase_index:min(len(tokens), phrase_index + args.len)])
                    for j in range(phrase_index, min(len(tokens), phrase_index + args.len)):
                        tokens.pop(phrase_index)
                    if phrase_index > 0:
                        words.append(" ".join(tokens[0:phrase_index]))
                    if phrase_index + 1 < len(tokens):
                        words.append(" ".join(tokens[phrase_index:]))
                    choices[target.find(choice)] = choice

                    # mask out with spaces
                    target = target.replace(choice, " " * len(choice), 1)

                for key in sorted(choices.keys()):
                    add_constraint(choices[key])

        print(source, *constraints, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', '-n', type=int, default=1, help="number of phrases")
    parser.add_argument('--len', '-l', type=int, default=1, help="phrase length")
    parser.add_argument('--add-sos', default=False, action='store_true', help='add <s> token')
    parser.add_argument('--add-eos', default=False, action='store_true', help='add </s> token')
    parser.add_argument('--seed', "-s", default=0, type=int)
    args = parser.parse_args()

    main(args)
