# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from tests.speech import TestFairseqSpeech


class TestXMTransformer(TestFairseqSpeech):
    def setUp(self):
        self.set_up_sotasty_es_en()

    # TODO: investigate increases BLEU score (30.42 -> 31.74)
    def test_sotasty_es_en_600m_checkpoint(self):
        self.base_test(
            ckpt_name="xm_transformer_600m_es_en_md.pt",
            reference_score=31.74,
            score_delta=0.2,
            max_tokens=3_000_000,
            max_positions=(1_000_000, 1_024),
            dataset="sotasty_es_en_test_ted",
            arg_overrides={"config_yaml": "cfg_es_en.yaml"},
            score_type="bleu",
        )


if __name__ == "__main__":
    unittest.main()
