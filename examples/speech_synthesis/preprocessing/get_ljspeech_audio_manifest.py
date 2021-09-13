# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from torchaudio.datasets import LJSPEECH
from tqdm import tqdm

from examples.speech_to_text.data_utils import save_df_to_tsv


log = logging.getLogger(__name__)

SPLITS = ["train", "dev", "test"]


def process(args):
    out_root = Path(args.output_data_root).absolute()
    out_root.mkdir(parents=True, exist_ok=True)

    # Generate TSV manifest
    print("Generating manifest...")
    # following FastSpeech's splits
    dataset = LJSPEECH(out_root.as_posix(), download=True)
    id_to_split = {}
    for x in dataset._flist:
        id_ = x[0]
        speaker = id_.split("-")[0]
        id_to_split[id_] = {
            "LJ001": "test", "LJ002": "test", "LJ003": "dev"
        }.get(speaker, "train")
    manifest_by_split = {split: defaultdict(list) for split in SPLITS}
    progress = tqdm(enumerate(dataset), total=len(dataset))
    for i, (waveform, _, utt, normalized_utt) in progress:
        sample_id = dataset._flist[i][0]
        split = id_to_split[sample_id]
        manifest_by_split[split]["id"].append(sample_id)
        audio_path = f"{dataset._path}/{sample_id}.wav"
        manifest_by_split[split]["audio"].append(audio_path)
        manifest_by_split[split]["n_frames"].append(len(waveform[0]))
        manifest_by_split[split]["tgt_text"].append(normalized_utt)
        manifest_by_split[split]["speaker"].append("ljspeech")
        manifest_by_split[split]["src_text"].append(utt)

    manifest_root = Path(args.output_manifest_root).absolute()
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest_by_split[split]),
            manifest_root / f"{split}.audio.tsv"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-root", "-d", required=True, type=str)
    parser.add_argument("--output-manifest-root", "-m", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
