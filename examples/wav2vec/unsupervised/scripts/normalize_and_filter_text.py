#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fasttext as ft
import os
import regex
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="reads text from stdin and outputs normalized, lid-filtered version to stdout"
    )
    parser.add_argument(
        "--fasttext-model",
        help="path to fasttext model",
        default="lid.187.bin",
    )
    parser.add_argument("--lang", help="language id", required=True)
    parser.add_argument(
        "--lid-threshold",
        type=float,
        help="threshold for this lang id probability",
        default=0.4,
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")

    lg = args.lang.lower()
    lg_label = f"__label__{lg}"
    thresh = args.lid_threshold

    if os.path.exists(args.fasttext_model):
        model = ft.load_model(args.fasttext_model)
    else:
        print(
            f"fasttext language id model {args.fasttext_model} not found. Proceeding without language filtering. "
            f"To enable language filtering, please download the latest language id model "
            f"from https://fasttext.cc/docs/en/language-identification.html",
            file=sys.stderr,
        )
        model = None

    for line in sys.stdin:
        line = line.strip()
        line = filter_r.sub(" ", line)
        line = " ".join(line.split())

        if model is not None:
            lid, prob = model.predict(line, k=100)
            try:
                target_idx = lid.index(lg_label)
            except ValueError:
                continue
            if target_idx == 0 or prob[target_idx] >= thresh:
                print(line)
        else:
            print(line)


if __name__ == "__main__":
    main()
