# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from tests.speech import TestFairseqSpeech

S3_BASE_URL = "https://dl.fbaipublicfiles.com/fairseq/"


class TestConvtransformerSimulTrans(TestFairseqSpeech):
    def setUp(self):
        self._set_up(
            "simul",
            "speech_tests/simul",
            ["config_gcmvn_specaug.yaml", "dict.txt", "dev.tsv"],
        )

    def test_waitk_checkpoint(self):
        """Only test model loading since fairseq currently doesn't support inference of simultaneous models"""
        _, _, _, _ = self.download_and_load_checkpoint(
            "checkpoint_best.pt",
            arg_overrides={"config_yaml": "config_gcmvn_specaug.yaml"},
        )
        return


if __name__ == "__main__":
    unittest.main()
