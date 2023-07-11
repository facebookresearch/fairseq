# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import os
import re
import unittest
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.scoring.wer import WerScorer
from fairseq.scoring.bleu import SacrebleuScorer
from fairseq import utils
import zipfile

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
        self.base_url = (
            s3_dir if re.search("^https:", s3_dir) else f"{S3_BASE_URL}/{s3_dir}"
        )
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

    def set_up_mustc_de_fbank(self):
        self._set_up(
            "mustc_de_fbank",
            "https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de",
            [
                "config.yaml",
                "spm.model",
                "dict.txt",
                "src_dict.txt",
                "tgt_dict.txt",
                "tst-COMMON.tsv",
                "tst-COMMON.zip",
            ],
        )

    def download_and_load_checkpoint(
        self,
        checkpoint_filename: str,
        arg_overrides: Optional[Dict[str, str]] = None,
        strict: bool = True,
    ):
        path = self.download(self.base_url, self.root, checkpoint_filename)
        _arg_overrides = arg_overrides or {}
        _arg_overrides["data"] = self.root.as_posix()
        models, cfg, task = load_model_ensemble_and_task(
            [path.as_posix()], arg_overrides=_arg_overrides, strict=strict
        )
        if self.use_cuda:
            for model in models:
                model.cuda()

        return models, cfg, task, self.build_generator(task, models, cfg)

    def build_generator(
        self,
        task,
        models,
        cfg,
    ):
        return task.build_generator(models, cfg)

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

    @torch.no_grad()
    def base_test(
        self,
        ckpt_name,
        reference_score,
        score_delta=0.3,
        dataset="librispeech_test-other",
        max_tokens=65_536,
        max_positions=(4_096, 1_024),
        arg_overrides=None,
        strict=True,
        score_type="wer",
    ):
        models, _, task, generator = self.download_and_load_checkpoint(
            ckpt_name, arg_overrides=arg_overrides, strict=strict
        )
        if not self.use_cuda:
            return

        batch_iterator = self.get_batch_iterator(
            task, dataset, max_tokens, max_positions
        )
        if score_type == "bleu":
            scorer = self.get_bleu_scorer()
        elif score_type == "wer":
            scorer = self.get_wer_scorer()
        else:
            raise Exception(f"Unsupported score type {score_type}")

        progress = tqdm(enumerate(batch_iterator), total=len(batch_iterator))
        for batch_idx, sample in progress:
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            hypo = task.inference_step(generator, models, sample)
            for i, sample_id in enumerate(sample["id"].tolist()):
                tgt_str, hypo_str = self.postprocess_tokens(
                    task,
                    sample["target"][i, :],
                    hypo[i][0]["tokens"].int().cpu(),
                )
                if batch_idx == 0 and i < 3:
                    print(f"T-{sample_id} {tgt_str}")
                    print(f"H-{sample_id} {hypo_str}")
                scorer.add_string(tgt_str, hypo_str)

        print(scorer.result_string() + f" (reference: {reference_score})")
        self.assertAlmostEqual(scorer.score(), reference_score, delta=score_delta)

    def postprocess_tokens(self, task, target, hypo_tokens):
        tgt_tokens = utils.strip_pad(target, task.tgt_dict.pad()).int().cpu()
        tgt_str = task.tgt_dict.string(tgt_tokens, "sentencepiece")
        hypo_str = task.tgt_dict.string(hypo_tokens, "sentencepiece")
        return tgt_str, hypo_str

    def unzip_files(self, zip_file_name):
        zip_file_path = self.root / zip_file_name
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.root / zip_file_name.strip(".zip"))
