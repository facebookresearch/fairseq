#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

for idx, line in enumerate(sys.stdin):
    print(f"utt{idx:010d} {line}", end="")
