#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Split a large file into shards while respecting document boundaries. Documents
should be separated by a single empty line.
"""

import argparse
import contextlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--num-shards', type=int)
    args = parser.parse_args()

    assert args.num_shards is not None and args.num_shards > 1

    with open(args.input, 'r', encoding='utf-8') as h:
        with contextlib.ExitStack() as stack:
            outputs = [
                stack.enter_context(open(args.input + ".shard" + str(i), "w", encoding="utf-8"))
                for i in range(args.num_shards)
            ]

            doc = []
            first_doc = [True]*args.num_shards

            def output_doc(i):
                if not first_doc[i]:
                    outputs[i].write("\n")
                first_doc[i] = False
                for line in doc:
                    outputs[i].write(line)
                doc.clear()

            num_docs = 0
            for line in h:
                if line.strip() == "":  # empty line indicates new document
                    output_doc(num_docs % args.num_shards)
                    num_docs += 1
                else:
                    doc.append(line)
            output_doc(num_docs % args.num_shards)


if __name__ == '__main__':
    main()
