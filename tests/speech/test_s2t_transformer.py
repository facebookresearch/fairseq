# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from tests.speech import TestFairseqSpeech


class TestS2TTransformer(TestFairseqSpeech):
    def setUp(self):
        self.set_up_librispeech()

    def test_librispeech_s2t_transformer_s_checkpoint(self):
        self.librispeech_s2t_test_base("librispeech_transformer_s.pt", 9)


if __name__ == "__main__":
    unittest.main()
