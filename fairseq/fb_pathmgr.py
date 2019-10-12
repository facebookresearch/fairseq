# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fvcore.common.file_io import PathManager
from fvcore.fb.manifold import ManifoldPathHandler
import os


class fb_pathmgr:

    @staticmethod
    def register():
        PathManager.register_handler(ManifoldPathHandler(max_parallel=16, timeout_sec=1800))

    @staticmethod
    def open(path: str, mode: str):
        dir_p = os.path.dirname(path)
        if dir_p != "":
            PathManager.mkdirs(dir_p)
        return PathManager.open(path, mode)

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False):
        return PathManager.copy(src_path, dst_path, overwrite)

    @staticmethod
    def isfile(path: str):
        return PathManager.isfile(path)

    @staticmethod
    def exists(path: str):
        return PathManager.exists(path)
