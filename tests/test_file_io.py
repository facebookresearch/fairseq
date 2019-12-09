# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import tempfile
import os
import shutil

from typing import Optional

import unittest
from unittest.mock import MagicMock


class TestFileIO(unittest.TestCase):

    _tmpdir: Optional[str] = None
    _tmpfile: Optional[str] = None
    _tmpfile_contents = "Hello, World"

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        with open(os.path.join(cls._tmpdir, "test.txt"), "w") as f:
            cls._tmpfile = f.name
            f.write(cls._tmpfile_contents)
            f.flush()

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_file_io(self):
        from fairseq.file_io import PathManager
        with PathManager.open(os.path.join(self._tmpdir, "test.txt"), "r") as f:
            s = f.read()
        self.assertEqual(s, self._tmpfile_contents)

    def test_file_io_oss(self):
        # Mock fvcore to simulate oss environment.
        sys.modules['fvcore'] = MagicMock()
        from fairseq.file_io import PathManager
        with PathManager.open(os.path.join(self._tmpdir, "test.txt"), "r") as f:
            s = f.read()
        self.assertEqual(s, self._tmpfile_contents)
