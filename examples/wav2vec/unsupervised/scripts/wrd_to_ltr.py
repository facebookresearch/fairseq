#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys


def main():
    for line in sys.stdin:
        print(" ".join(list(line.strip().replace(" ", "|"))) + " |")


if __name__ == "__main__":
    main()
