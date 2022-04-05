# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
from torchaudio.datasets import VCTK
from tqdm import tqdm

from examples.speech_to_text.data_utils import save_df_to_tsv


log = logging.getLogger(__name__)

SPLITS = ["train", "dev", "test"]


def normalize_text(text):
    return re.sub(r"[^a-zA-Z.?!,'\- ]", '', text)


def process(args):
    out_root = Path(args.output_data_root).absolute()
    out_root.mkdir(parents=True, exist_ok=True)

    # Generate TSV manifest
    print("Generating manifest...")
    dataset = VCTK(out_root.as_posix(), download=False)
    ids = list(dataset._walker)
    np.random.seed(args.seed)
    np.random.shuffle(ids)
    n_train = len(ids) - args.n_dev - args.n_test
    _split = ["train"] * n_train + ["dev"] * args.n_dev + ["test"] * args.n_test
    id_to_split = dict(zip(ids, _split))
    manifest_by_split = {split: defaultdict(list) for split in SPLITS}
    progress = tqdm(enumerate(dataset), total=len(dataset))
    for i, (waveform, _, text, speaker_id, _) in progress:
        sample_id = dataset._walker[i]
        _split = id_to_split[sample_id]
        audio_dir = Path(dataset._path) / dataset._folder_audio / speaker_id
        audio_path = audio_dir / f"{sample_id}.wav"
        text = normalize_text(text)
        manifest_by_split[_split]["id"].append(sample_id)
        manifest_by_split[_split]["audio"].append(audio_path.as_posix())
        manifest_by_split[_split]["n_frames"].append(len(waveform[0]))
        manifest_by_split[_split]["tgt_text"].append(text)
        manifest_by_split[_split]["speaker"].append(speaker_id)
        manifest_by_split[_split]["src_text"].append(text)

    manifest_root = Path(args.output_manifest_root).absolute()
    manifest_root.mkdir(parents=True, exist_ok=True)
    for _split in SPLITS:
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest_by_split[_split]),
            manifest_root / f"{_split}.audio.tsv"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-root", "-d", required=True, type=str)
    parser.add_argument("--output-manifest-root", "-m", required=True, type=str)
    parser.add_argument("--n-dev", default=50, type=int)
    parser.add_argument("--n-test", default=100, type=int)
    parser.add_argument("--seed", "-s", default=1234, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
