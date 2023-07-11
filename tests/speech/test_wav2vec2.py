# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from tests.speech import TestFairseqSpeech
from fairseq.data.data_utils import post_process
from fairseq import utils
from omegaconf import open_dict

S3_BASE_URL = "https://dl.fbaipublicfiles.com/fairseq"


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestWav2Vec2(TestFairseqSpeech):
    def setUp(self):
        self._set_up(
            "librispeech_w2v2",
            "conformer/wav2vec2/librispeech",
            [
                "test_librispeech-other.ltr",
                "test_librispeech-other.tsv",
                "test_librispeech-other_small.ltr_100",
                "test_librispeech-other_small.tsv",
                "test-other.zip",
                "dict.ltr.txt",
                "dict.ltr_100.txt",
            ],
        )
        self.unzip_files(
            "test-other.zip",
        )

    def test_transformer_w2v2(self):
        self.base_test(
            ckpt_name="transformer_oss_small_100h.pt",
            reference_score=38,
            score_delta=1,
            dataset="test_librispeech-other",
            max_tokens=1000000,
            max_positions=(700000, 1000),
            arg_overrides={
                "task": "audio_finetuning",
                "labels": "ltr",
                "nbest": 1,
                "tpu": False,
            },
            strict=False,
        )

    def test_conformer_w2v2(self):
        self.base_test(
            ckpt_name="conformer_LS_PT_LS_FT_rope.pt",
            reference_score=4.5,
            score_delta=1,
            dataset="test_librispeech-other_small",
            max_tokens=1000000,
            max_positions=(700000, 1000),
            arg_overrides={
                "task": "audio_finetuning",
                "labels": "ltr_100",
                "nbest": 1,
                "tpu": False,
            },
            strict=True,
        )

    def build_generator(self, task, models, cfg):
        try:
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
        except Exception:
            raise Exception("Cannot run this test without flashlight dependency")
        with open_dict(cfg):
            cfg.nbest = 1
        return W2lViterbiDecoder(cfg, task.target_dictionary)

    def postprocess_tokens(self, task, target, hypo_tokens):
        tgt_tokens = utils.strip_pad(target, task.target_dictionary.pad()).int().cpu()
        tgt_str = task.target_dictionary.string(tgt_tokens)
        tgt_str = post_process(tgt_str, "letter")

        hypo_pieces = task.target_dictionary.string(hypo_tokens)
        hypo_str = post_process(hypo_pieces, "letter")
        return tgt_str, hypo_str


if __name__ == "__main__":
    unittest.main()
