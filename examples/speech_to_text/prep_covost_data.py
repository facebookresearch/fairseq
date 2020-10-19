#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import logging
import os
import os.path as op
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    CV_URL_TEMPLATE = (
        "https://voice-prod-bundler-ee1969a6ce8178826482b88"
        "e843c335139bd3fb4.s3.amazonaws.com/{ver}/{lang}.tar.gz"
    )
    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]

    CV_VERSION_ID = {1: "cv-corpus-3", 2: "cv-corpus-4-2019-12-10"}

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
        download: bool = False,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root = os.path.join(root, "raw")
        os.makedirs(self.root, exist_ok=True)

        cv_url = self.CV_URL_TEMPLATE.format(
            ver=self.CV_VERSION_ID[version], lang=source_language
        )
        cv_archive = os.path.join(self.root, os.path.basename(cv_url))
        if download:
            if not os.path.isfile(cv_archive):
                download_url(cv_url, self.root, hash_value=None)
            extract_archive(cv_archive)

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        covost_archive = os.path.join(self.root, os.path.basename(covost_url))
        if download:
            if not os.path.isfile(covost_archive):
                download_url(covost_url, self.root, hash_value=None)
            extract_archive(covost_archive)

        cv_tsv = self.load_from_tsv(os.path.join(self.root, "validated.tsv"))
        covost_tsv = self.load_from_tsv(
            os.path.join(self.root, os.path.basename(covost_url).replace(".tar.gz", ""))
        )
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        self.data = df.to_dict(orient="index").items()
        self.data = [v for k, v in sorted(self.data, key=lambda x: x[0])]

    @classmethod
    def load_from_tsv(cls, path: str):
        return pd.read_csv(
            path,
            sep="\t",
            header=0,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
            na_filter=False,
        )

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = os.path.join(self.root, "clips", data["path"])
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = op.join(args.data_root, args.src_lang)
    os.makedirs(root, exist_ok=True)
    # Extract features
    feature_root = op.join(root, "fbank80")
    os.makedirs(feature_root, exist_ok=True)
    for split in CoVoST.SPLITS:
        print(f"Fetching split {split}...")
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang, download=True)
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
            extract_fbank_features(
                waveform, sample_rate, op.join(feature_root, f"{utt_id}.npy")
            )
    # Pack features into ZIP
    zip_filename = "fbank80.zip"
    zip_path = op.join(root, zip_filename)
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(args.data_root, f"{args.src_lang}/{zip_filename}")
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in CoVoST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(zip_manifest[utt_id])
            duration_ms = int(wav.size(1) / sr * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
            manifest["speaker"].append(speaker_id)
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, op.join(root, f"{split}_{task}.tsv"))
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            f.name, op.join(root, spm_filename_prefix), args.vocab_type, args.vocab_size
        )
    # Generate config YAML
    gen_config_yaml(
        root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{task}.yaml",
        specaugment_policy="lb",
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
