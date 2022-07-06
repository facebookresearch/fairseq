# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import struct
import sys

assert len(sys.argv) == 2

with open(sys.argv[1], "rb") as f:
    assert f.read(9) == b"MMIDIDX\x00\x00"
    _ = f.read(9)
    print(struct.unpack("<Q", f.read(8))[0])
