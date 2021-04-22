# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import combinations


def read_translations(path, n_repeats):
    segment_counter = 0
    segment_translations = []
    translations = defaultdict(list)
    for line in open(path):
        segment_translations.append(" ".join(line.split()))
        if len(segment_translations) == n_repeats:
            translations[segment_counter] = segment_translations
            segment_translations = []
            segment_counter += 1
    return translations


def generate_input(translations, n_repeats):
    _, ref_path = tempfile.mkstemp()
    _, mt_path = tempfile.mkstemp()
    ref_fh = open(ref_path, "w")
    mt_fh = open(mt_path, "w")
    for segid in sorted(translations.keys()):
        assert len(translations[segid]) == n_repeats
        indexes = combinations(range(n_repeats), 2)
        for idx1, idx2 in indexes:
            mt_fh.write(translations[segid][idx1].strip() + "\n")
            ref_fh.write(translations[segid][idx2].strip() + "\n")
    sys.stderr.write("\nSaved translations to %s and %s" % (ref_path, mt_path))
    return ref_path, mt_path


def run_meteor(ref_path, mt_path, metric_path, lang="en"):
    _, out_path = tempfile.mkstemp()
    subprocess.call(
        [
            "java",
            "-Xmx2G",
            "-jar",
            metric_path,
            mt_path,
            ref_path,
            "-p",
            "0.5 0.2 0.6 0.75",  # default parameters, only changed alpha to give equal weight to P and R
            "-norm",
            "-l",
            lang,
        ],
        stdout=open(out_path, "w"),
    )
    os.remove(ref_path)
    os.remove(mt_path)
    sys.stderr.write("\nSaved Meteor output to %s" % out_path)
    return out_path


def read_output(meteor_output_path, n_repeats):
    n_combinations = math.factorial(n_repeats) / (
        math.factorial(2) * math.factorial(n_repeats - 2)
    )
    raw_scores = []
    average_scores = []
    for line in open(meteor_output_path):
        if not line.startswith("Segment "):
            continue
        score = float(line.strip().split("\t")[1])
        raw_scores.append(score)
        if len(raw_scores) == n_combinations:
            average_scores.append(sum(raw_scores) / n_combinations)
            raw_scores = []
    os.remove(meteor_output_path)
    return average_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile")
    parser.add_argument("-n", "--repeat_times", type=int)
    parser.add_argument("-m", "--meteor")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    translations = read_translations(args.infile, args.repeat_times)
    sys.stderr.write("\nGenerating input for Meteor...")
    ref_path, mt_path = generate_input(translations, args.repeat_times)
    sys.stderr.write("\nRunning Meteor...")
    out_path = run_meteor(ref_path, mt_path, args.meteor)
    sys.stderr.write("\nReading output...")
    scores = read_output(out_path, args.repeat_times)
    sys.stderr.write("\nWriting results...")
    with open(args.output, "w") as o:
        for scr in scores:
            o.write("{}\n".format(scr))
    o.close()


if __name__ == "__main__":
    main()
