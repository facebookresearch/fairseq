#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import regex
import sys


def main():
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")

    for line in sys.stdin:
        line = line.strip()
        line = filter_r.sub(" ", line)
        line = " ".join(line.split())
        print(line)


if __name__ == "__main__":
    main()
