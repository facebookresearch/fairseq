# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from typing import Optional


class TestFileChunker(unittest.TestCase):
    _tmpdir: Optional[str] = None
    _tmpfile: Optional[str] = None
    _line_content = "Hello, World\n"
    _num_bytes = None
    _num_lines = 200
    _num_splits = 20

    @classmethod
    def setUpClass(cls) -> None:
        cls._num_bytes = len(cls._line_content.encode("utf-8"))
        cls._tmpdir = tempfile.mkdtemp()
        with open(os.path.join(cls._tmpdir, "test.txt"), "w") as f:
            cls._tmpfile = f.name
            for _i in range(cls._num_lines):
                f.write(cls._line_content)
            f.flush()

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_find_offsets(self):
        from fairseq.file_chunker_utils import find_offsets

        offsets = find_offsets(self._tmpfile, self._num_splits)
        self.assertEqual(len(offsets), self._num_splits + 1)
        (zero, *real_offsets, last) = offsets
        self.assertEqual(zero, 0)
        for i, o in enumerate(real_offsets):
            self.assertEqual(
                o,
                self._num_bytes
                + ((i + 1) * self._num_bytes * self._num_lines / self._num_splits),
            )
        self.assertEqual(last, self._num_bytes * self._num_lines)

    def test_readchunks(self):
        from fairseq.file_chunker_utils import Chunker, find_offsets

        offsets = find_offsets(self._tmpfile, self._num_splits)
        for start, end in zip(offsets, offsets[1:]):
            with Chunker(self._tmpfile, start, end) as lines:
                all_lines = list(lines)
                num_lines = self._num_lines / self._num_splits
                self.assertAlmostEqual(
                    len(all_lines), num_lines, delta=1
                )  # because we split on the bites, we might end up with one more/less line in a chunk
                self.assertListEqual(
                    all_lines, [self._line_content for _ in range(len(all_lines))]
                )
