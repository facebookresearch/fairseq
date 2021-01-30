# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock


class TestIOPath(unittest.TestCase):

    def test_no_iopath(self):
        from .test_reproducibility import TestReproducibility

        with mock.patch.dict("sys.modules", {"iopath": None}):
            # reuse reproducibility tests, which are e2e tests that should cover
            # most checkpoint related functionality
            TestReproducibility._test_reproducibility(self, "test_reproducibility")

    def test_no_supports_rename(self):
        from .test_reproducibility import TestReproducibility

        with mock.patch("fairseq.file_io.PathManager.supports_rename") as mock_fn:
            mock_fn.return_value = False
            TestReproducibility._test_reproducibility(self, "test_reproducibility")


if __name__ == "__main__":
    unittest.main()
