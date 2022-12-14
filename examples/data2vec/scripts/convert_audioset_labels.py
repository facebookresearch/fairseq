#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description="convert audioset labels")
    # fmt: off
    parser.add_argument('in_file', help='audioset csv file to convert')
    parser.add_argument('--manifest', required=True, metavar='PATH', help='wav2vec-like manifest')
    parser.add_argument('--descriptors', required=True, metavar='PATH', help='path to label descriptor file')
    parser.add_argument('--output', required=True, metavar='PATH', help='where to output converted labels')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    label_descriptors = {}
    with open(args.descriptors, "r") as ldf:
        next(ldf)
        for line in ldf:
            if line.strip() == "":
                continue

            items = line.split(",")
            assert len(items) > 2, line
            idx = items[0]
            lbl = items[1]
            assert lbl not in label_descriptors, lbl
            label_descriptors[lbl] = idx

    labels = {}
    with open(args.in_file, "r") as ifd:
        for line in ifd:
            if line.lstrip().startswith("#"):
                continue
            items = line.rstrip().split(",")
            id = items[0].strip()
            start = items[1].strip()
            end = items[2].strip()
            lbls = [label_descriptors[it.strip(' "')] for it in items[3:]]
            labels[id] = [start, end, ",".join(lbls)]

    with open(args.manifest, "r") as mf, open(args.output, "w") as of:
        next(mf)
        for line in mf:
            path, _ = line.split("\t")
            id = os.path.splitext(os.path.basename(path))[0]
            lbl = labels[id]
            print("\t".join(lbl), file=of)


if __name__ == "__main__":
    main()
