# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

import numpy as np


aggregate_funcs = {
    "std": np.std,
    "var": np.var,
    "median": np.median,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, type=str)
    parser.add_argument("-n", "--repeat_times", required=True, type=int)
    parser.add_argument("-o", "--output_file", required=False)
    parser.add_argument("-f", "--func", required=False, default="mean")
    args = parser.parse_args()

    stream = open(args.output_file, "w") if args.output_file else sys.stdout

    segment_scores = []
    for line in open(args.input_file):
        segment_scores.append(float(line.strip()))
        if len(segment_scores) == args.repeat_times:
            stream.write("{}\n".format(aggregate_funcs[args.func](segment_scores)))
            segment_scores = []


if __name__ == "__main__":
    main()
