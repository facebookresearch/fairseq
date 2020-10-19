#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.average_checkpoints import main
from fairseq.file_io import PathManager


# support fb specific path mananger
try:
    from fvcore.fb.manifold import ManifoldPathHandler

    PathManager.register_handler(ManifoldPathHandler(max_parallel=16, timeout_sec=1800))
except Exception:
    pass


if __name__ == "__main__":
    main()
