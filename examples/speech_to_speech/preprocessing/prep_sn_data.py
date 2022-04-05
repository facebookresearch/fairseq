#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from examples/wav2vec/wav2vec_manifest.py
"""
Data preparation for the speech normalizer
"""

import argparse
import glob
import os

import soundfile

from examples.speech_to_speech.preprocessing.data_utils import load_units, process_units


def process(args):
    assert (
        args.for_inference or args.target_unit is not None
    ), "missing --target-unit or --for-inference"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dir_path = os.path.realpath(args.audio_dir)
    search_path = os.path.join(dir_path, "**/*." + args.ext)

    if args.target_unit:
        unit_data = load_units(args.target_unit)

    with open(os.path.join(args.output_dir, f"{args.data_name}.tsv"), "w") as o_t, open(
        os.path.join(args.output_dir, f"{args.data_name}.unit"), "w"
    ) as o_u:
        print(dir_path, file=o_t)
        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)
            frames = soundfile.info(fname).frames
            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=o_t
            )

            if args.for_inference:
                print("0", file=o_u)
            else:
                sample_id = os.path.basename(file_path)[: -len(args.ext) - 1]
                assert (
                    sample_id in unit_data
                ), f'{fname} does not have unit data in {args.target_unit}. Expecting sample_id "{sample_id}".'
                target_units = process_units(unit_data[sample_id], reduce=True)
                print(" ".join(target_units), file=o_u)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, type=str, help="audio directory")
    parser.add_argument("--ext", default="flac", type=str, help="audio extension")
    parser.add_argument(
        "--data-name",
        required=True,
        type=str,
        help="dataset name",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="output directory"
    )
    parser.add_argument(
        "--for-inference",
        action="store_true",
        help="set this if preparing data for running inference with a speech normalizer",
    )
    parser.add_argument(
        "--target-unit",
        default=None,
        type=str,
        help="a file containing unit sequences in the format: sample_id|u1 u2 ...",
    )

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
