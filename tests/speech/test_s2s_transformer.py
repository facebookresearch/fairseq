# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from tests.speech import TestFairseqSpeech
from fairseq import utils

S3_BASE_URL = "https://dl.fbaipublicfiles.com/fairseq/"


class TestS2STransformer(TestFairseqSpeech):
    def setUp(self):
        self._set_up(
            "s2s",
            "speech_tests/s2s",
            [
                "dev_shuf200.tsv",
                "src_feat.zip",
                "config_specaug_lb.yaml",
                "config_letter_enc_unigram_dec.yaml",
            ],
        )

    def test_s2s_transformer_checkpoint(self):
        self.base_test(
            ckpt_name="s2u_transformer_reduced_fisher.pt",
            reference_score=38.3,
            dataset="dev_shuf200",
            arg_overrides={
                "config_yaml": "config_specaug_lb.yaml",
                "target_is_code": True,
                "target_code_size": 100,
            },
            score_type="bleu",
        )

    def postprocess_tokens(self, task, target, hypo_tokens):
        tgt_tokens = utils.strip_pad(target, task.tgt_dict.pad()).int().cpu()
        tgt_str = task.tgt_dict.string(tgt_tokens)
        hypo_str = task.tgt_dict.string(hypo_tokens)
        return tgt_str, hypo_str


if __name__ == "__main__":
    unittest.main()
