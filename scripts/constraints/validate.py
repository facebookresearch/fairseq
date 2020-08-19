#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

"""Reads in a fairseq output file, and verifies that the constraints
(C- lines) are present in the output (the first H- line). Assumes that
constraints are listed prior to the first hypothesis.
"""

constraints = []
found = 0
total = 0
for line in sys.stdin:
    if line.startswith("C-"):
        constraints.append(line.rstrip().split("\t")[1])
    elif line.startswith("H-"):
        text = line.split("\t")[2]

        for constraint in constraints:
            total += 1
            if constraint in text:
                found += 1
            else:
                print(f"No {constraint} in {text}", file=sys.stderr)

        constraints = []

print(f"Found {found} / {total} = {100 * found / total:.1f}%")
