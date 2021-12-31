# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import os
import unittest
from pathlib import Path
from typing import List, Dict, Optional

import torch

from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.scoring.wer import WerScorer
from fairseq.scoring.bleu import SacrebleuScorer

S3_BASE_URL = "https://dl.fbaipublicfiles.com/fairseq"


class TestFairseqSpeech(unittest.TestCase):
    @classmethod
    def download(cls, base_url: str, out_root: Path, filename: str):
        url = f"{base_url}/{filename}"
        path = out_root / filename
        if not path.exists():
            torch.hub.download_url_to_file(url, path.as_posix(), progress=True)
        return path

    def _set_up(self, dataset_id: str, s3_dir: str, data_filenames: List[str]):
        self.use_cuda = torch.cuda.is_available()
        self.root = Path.home() / ".cache" / "fairseq" / dataset_id
        self.root.mkdir(exist_ok=True, parents=True)
        os.chdir(self.root)
        self.base_url = f"{S3_BASE_URL}/{s3_dir}"
        for filename in data_filenames:
            self.download(self.base_url, self.root, filename)

    def set_up_librispeech(self):
        self._set_up(
            "librispeech",
            "s2t/librispeech",
            [
                "cfg_librispeech.yaml",
                "spm_librispeech_unigram10000.model",
                "spm_librispeech_unigram10000.txt",
                "librispeech_test-other.tsv",
                "librispeech_test-other.zip",
            ],
        )

    def set_up_ljspeech(self):
        self._set_up(
            "ljspeech",
            "s2/ljspeech",
            [
                "cfg_ljspeech_g2p.yaml",
                "ljspeech_g2p_gcmvn_stats.npz",
                "ljspeech_g2p.txt",
                "ljspeech_test.tsv",
                "ljspeech_test.zip",
            ],
        )

    def set_up_sotasty_es_en(self):
        self._set_up(
            "sotasty_es_en",
            "s2t/big/es-en",
            [
                "cfg_es_en.yaml",
                "spm_bpe32768_es_en.model",
                "spm_bpe32768_es_en.txt",
                "sotasty_es_en_test_ted.tsv",
                "sotasty_es_en_test_ted.zip",
            ],
        )

    def download_and_load_checkpoint(
        self, checkpoint_filename: str, arg_overrides: Optional[Dict[str, str]] = None
    ):
        path = self.download(self.base_url, self.root, checkpoint_filename)
        _arg_overrides = arg_overrides or {}
        _arg_overrides["data"] = self.root.as_posix()
        models, cfg, task = load_model_ensemble_and_task(
            [path.as_posix()], arg_overrides=_arg_overrides
        )
        if self.use_cuda:
            for model in models:
                model.cuda()
        generator = task.build_generator(models, cfg)
        return models, cfg, task, generator

    @classmethod
    def get_batch_iterator(cls, task, test_split, max_tokens, max_positions):
        task.load_dataset(test_split)
        return task.get_batch_iterator(
            dataset=task.dataset(test_split),
            max_tokens=max_tokens,
            max_positions=max_positions,
            num_workers=1,
        ).next_epoch_itr(shuffle=False)

    @classmethod
    def get_wer_scorer(
        cls, tokenizer="none", lowercase=False, remove_punct=False, char_level=False
    ):
        scorer_args = {
            "wer_tokenizer": tokenizer,
            "wer_lowercase": lowercase,
            "wer_remove_punct": remove_punct,
            "wer_char_level": char_level,
        }
        return WerScorer(Namespace(**scorer_args))

    @classmethod
    def get_bleu_scorer(cls, tokenizer="13a", lowercase=False, char_level=False):
        scorer_args = {
            "sacrebleu_tokenizer": tokenizer,
            "sacrebleu_lowercase": lowercase,
            "sacrebleu_char_level": char_level,
        }
        return SacrebleuScorer(Namespace(**scorer_args))
