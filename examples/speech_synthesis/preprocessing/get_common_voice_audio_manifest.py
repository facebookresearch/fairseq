# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import torchaudio
from tqdm import tqdm

from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv


log = logging.getLogger(__name__)

SPLITS = ["train", "dev", "test"]


def get_top_n(
        root: Path, n_speakers: int = 10, min_n_tokens: int = 5
) -> pd.DataFrame:
    df = load_df_from_tsv(root / "validated.tsv")
    df["n_tokens"] = [len(s.split()) for s in df["sentence"]]
    df = df[df["n_tokens"] >= min_n_tokens]
    df["n_frames"] = [
        torchaudio.info((root / "clips" / p).as_posix()).num_frames
        for p in tqdm(df["path"])
    ]
    df["id"] = [Path(p).stem for p in df["path"]]
    total_duration_ms = df.groupby("client_id")["n_frames"].agg(["sum"])
    total_duration_ms = total_duration_ms.sort_values("sum", ascending=False)

    top_n_total_duration_ms = total_duration_ms.head(n_speakers)
    top_n_client_ids = set(top_n_total_duration_ms.index.tolist())
    df_top_n = df[df["client_id"].isin(top_n_client_ids)]
    return df_top_n


def get_splits(
        df, train_split_ratio=0.99, speaker_in_all_splits=False, rand_seed=0
) -> Tuple[Dict[str, str], List[str]]:
    np.random.seed(rand_seed)
    dev_split_ratio = (1. - train_split_ratio) / 3
    grouped = list(df.groupby("client_id"))
    id_to_split = {}
    for _, cur_df in tqdm(grouped):
        cur_n_examples = len(cur_df)
        if speaker_in_all_splits and cur_n_examples < 3:
            continue
        cur_n_train = int(cur_n_examples * train_split_ratio)
        cur_n_dev = int(cur_n_examples * dev_split_ratio)
        cur_n_test = cur_n_examples - cur_n_dev - cur_n_train
        if speaker_in_all_splits and cur_n_dev * cur_n_test == 0:
            cur_n_dev, cur_n_test = 1, 1
            cur_n_train = cur_n_examples - cur_n_dev - cur_n_test
        cur_indices = cur_df.index.tolist()
        cur_shuffled_indices = np.random.permutation(cur_n_examples)
        cur_shuffled_indices = [cur_indices[i] for i in cur_shuffled_indices]
        cur_indices_by_split = {
            "train": cur_shuffled_indices[:cur_n_train],
            "dev": cur_shuffled_indices[cur_n_train: cur_n_train + cur_n_dev],
            "test": cur_shuffled_indices[cur_n_train + cur_n_dev:]
        }
        for split in SPLITS:
            for i in cur_indices_by_split[split]:
                id_ = df["id"].loc[i]
                id_to_split[id_] = split
    return id_to_split, sorted(df["client_id"].unique())


def convert_to_wav(root: Path, filenames: List[str], target_sr=16_000):
    out_root = root / "wav"
    out_root.mkdir(exist_ok=True, parents=True)
    print("Converting to WAV...")
    for n in tqdm(filenames):
        in_path = (root / "clips" / n).as_posix()
        waveform, sr = torchaudio.load(in_path)
        converted, converted_sr = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sr, [["rate", str(target_sr)], ["channels", "1"]]
        )
        out_path = (out_root / Path(n).with_suffix(".wav").name).as_posix()
        torchaudio.save(out_path, converted, converted_sr, encoding="PCM_S",
                        bits_per_sample=16)


def process(args):
    data_root = Path(args.data_root).absolute() / args.lang

    # Generate TSV manifest
    print("Generating manifest...")

    df_top_n = get_top_n(data_root)
    id_to_split, speakers = get_splits(df_top_n)

    if args.convert_to_wav:
        convert_to_wav(data_root, df_top_n["path"].tolist())

    manifest_by_split = {split: defaultdict(list) for split in SPLITS}
    for sample in tqdm(df_top_n.to_dict(orient="index").values()):
        sample_id = sample["id"]
        split = id_to_split[sample_id]
        manifest_by_split[split]["id"].append(sample_id)
        if args.convert_to_wav:
            audio_path = data_root / "wav" / f"{sample_id}.wav"
        else:
            audio_path = data_root / "clips" / f"{sample_id}.mp3"
        manifest_by_split[split]["audio"].append(audio_path.as_posix())
        manifest_by_split[split]["n_frames"].append(sample["n_frames"])
        manifest_by_split[split]["tgt_text"].append(sample["sentence"])
        manifest_by_split[split]["speaker"].append(sample["client_id"])
        manifest_by_split[split]["src_text"].append(sample["sentence"])

    output_root = Path(args.output_manifest_root).absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest_by_split[split]),
            output_root / f"{split}.audio.tsv"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--output-manifest-root", "-m", required=True, type=str)
    parser.add_argument("--lang", "-l", required=True, type=str)
    parser.add_argument("--convert-to-wav", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
