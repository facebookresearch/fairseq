# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from pathlib import Path

import torch

S3_BASE_URL = "https://dl.fbaipublicfiles.com/fairseq"


class TestFairseqSpeech(unittest.TestCase):
    @classmethod
    def download(cls, base_url: str, out_root: Path, filename: str):
        url = f"{base_url}/{filename}"
        path = out_root / filename
        if not path.exists():
            torch.hub.download_url_to_file(url, path.as_posix(), progress=True)
        return path

    def set_up_librispeech(self):
        self.use_cuda = torch.cuda.is_available()
        self.root = Path.home() / ".cache" / "fairseq" / "librispeech"
        self.root.mkdir(exist_ok=True, parents=True)
        os.chdir(self.root)
        self.data_filenames = [
            "cfg_librispeech.yaml",
            "spm_librispeech_unigram10000.model",
            "spm_librispeech_unigram10000.txt",
            "librispeech_test-other.tsv",
            "librispeech_test-other.zip",
        ]
        self.base_url = f"{S3_BASE_URL}/s2t/librispeech"
        for filename in self.data_filenames:
            self.download(self.base_url, self.root, filename)

    def set_up_ljspeech(self):
        self.use_cuda = torch.cuda.is_available()
        self.root = Path.home() / ".cache" / "fairseq" / "ljspeech"
        self.root.mkdir(exist_ok=True, parents=True)
        os.chdir(self.root)
        self.data_filenames = [
            "cfg_ljspeech_g2p.yaml",
            "ljspeech_g2p_gcmvn_stats.npz",
            "ljspeech_g2p.txt",
            "ljspeech_test.tsv",
            "ljspeech_test.zip",
        ]
        self.base_url = f"{S3_BASE_URL}/s2/ljspeech"
        for filename in self.data_filenames:
            self.download(self.base_url, self.root, filename)
