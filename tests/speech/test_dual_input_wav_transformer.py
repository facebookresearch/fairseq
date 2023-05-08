# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import namedtuple
from pathlib import Path

import torch
from tqdm import tqdm

import fairseq
from fairseq import utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.scoring.bleu import SacrebleuScorer
from fairseq.tasks import import_tasks
from tests.speech import S3_BASE_URL, TestFairseqSpeech


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestLibrispeechDualInputWavTransformer(TestFairseqSpeech):
    def setUp(self):
        dataset_id = "librispeech_wvtrasnformer"
        base_url = "https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/finetuned"
        data_filenames = [
            "checkpoint_ave_10.pt",
            "spm.model",
            "src_dict.txt",
            "tgt_dict.txt",
            "config.yaml",
        ]
        self._set_up(
            dataset_id,
            "s2t",
            [
                "librispeech_flac_test-other.tsv",
                "librispeech_flac_test-other.zip",
            ],
        )
        for filename in data_filenames:
            self.download(base_url, self.root, filename)

    def import_user_module(self):
        user_dir = (
            Path(fairseq.__file__).parent.parent / "examples/speech_text_joint_to_text"
        )
        Arg = namedtuple("Arg", ["user_dir"])
        arg = Arg(user_dir.__str__())
        utils.import_user_module(arg)

    @torch.no_grad()
    def test_librispeech_dualinput_wav_transformer_checkpoint(self):
        self.import_user_module()
        checkpoint_filename = "checkpoint_ave_10.pt"
        arg_overrides = {
            "config_yaml": "config.yaml",
            "load_pretrained_speech_text_encoder": "",
            "load_pretrained_speech_text_decoder": "",
            "beam": 10,
            "nbest": 1,
            "lenpen": 1.0,
            "load_speech_only": True,
        }
        self.base_test(
            checkpoint_filename,
            4.6,
            dataset="librispeech_flac_test-other",
            max_tokens=800000,
            max_positions=(800000, 1024),
            arg_overrides=arg_overrides,
        )


if __name__ == "__main__":
    unittest.main()
