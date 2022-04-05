#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
import torchaudio

import soundfile as sf
from tqdm import tqdm
import pandas as pd

from examples.speech_synthesis.data_utils import extract_logmel_spectrogram
from examples.speech_to_speech.preprocessing.data_utils import gen_config_yaml
from examples.speech_to_text.data_utils import create_zip, get_zip_manifest, save_df_to_tsv
from fairseq.data.audio.audio_utils import convert_waveform


logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames"]


def prepare_target_data(args, tgt_audios):
    feature_name = "logmelspec80"
    zip_path = args.output_root / f"{feature_name}.zip"
    if zip_path.exists():
        print(f"{zip_path} exists.")
        return zip_path

    feature_root = args.output_root / feature_name
    feature_root.mkdir(exist_ok=True)

    print("Extracting Mel spectrogram features...")
    for tgt_audio in tqdm(tgt_audios):
        sample_id = tgt_audio.stem
        waveform, sample_rate = torchaudio.load(tgt_audio.as_posix())
        waveform, sample_rate = convert_waveform(
            waveform, sample_rate, normalize_volume=args.normalize_volume,
            to_sample_rate=args.sample_rate
        )
        extract_logmel_spectrogram(
            waveform, sample_rate, feature_root / f"{sample_id}.npy",
            win_length=args.win_length, hop_length=args.hop_length,
            n_fft=args.n_fft, n_mels=args.n_mels, f_min=args.f_min,
            f_max=args.f_max
        )
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    shutil.rmtree(feature_root)

    return zip_path


def process(args):
    os.makedirs(args.output_root, exist_ok=True)

    manifest = {}
    tgt_audios = []
    for split in args.data_split:
        print(f"Processing {split}...")

        manifest[split] = {c: [] for c in MANIFEST_COLUMNS}
        missing_tgt_audios = []
        src_audios = list(args.source_dir.glob(f"{split}/*.wav"))
        for src_audio in tqdm(src_audios):
            sample_id = src_audio.stem

            tgt_audio = args.target_dir / split / f"{sample_id}.wav"
            if not tgt_audio.is_file():
                missing_tgt_audios.append(sample_id)
                continue

            tgt_audios.append(tgt_audio)

            src_n_frames = sf.info(src_audio.as_posix()).frames
            manifest[split]["id"].append(sample_id)
            manifest[split]["src_audio"].append(src_audio.as_posix())
            manifest[split]["src_n_frames"].append(
                src_n_frames // 160
            )  # estimation of 10-ms frame for 16kHz audio

        print(f"Processed {len(manifest[split]['id'])} samples")
        if len(missing_tgt_audios) > 0:
            print(
                f"{len(missing_tgt_audios)} with missing target data (first 3 examples: {', '.join(missing_tgt_audios[:3])})"
            )

    # Extract features and pack features into ZIP
    zip_path = prepare_target_data(args, tgt_audios)

    print("Fetching ZIP manifest...")
    tgt_audio_paths, tgt_audio_lengths = get_zip_manifest(zip_path)

    print("Generating manifest...")
    for split in args.data_split:
        print(f"Processing {split}...")

        for sample_id in tqdm(manifest[split]["id"]):
            manifest[split]["tgt_audio"].append(tgt_audio_paths[sample_id])
            manifest[split]["tgt_n_frames"].append(tgt_audio_lengths[sample_id])

        out_manifest = args.output_root / f"{split}.tsv"
        print(f"Writing manifest to {out_manifest}...")
        save_df_to_tsv(pd.DataFrame.from_dict(manifest[split]), out_manifest)

    # Generate config YAML
    win_len_t = args.win_length / args.sample_rate
    hop_len_t = args.hop_length / args.sample_rate
    extra = {
        "features": {
            "type": "spectrogram+melscale+log",
            "sample_rate": args.sample_rate,
            "eps": 1e-5, "n_mels": args.n_mels, "n_fft": args.n_fft,
            "window_fn": "hann", "win_length": args.win_length,
            "hop_length": args.hop_length,
            "win_len_t": win_len_t, "hop_len_t": hop_len_t,
            "f_min": args.f_min, "f_max": args.f_max,
            "n_stft": args.n_fft // 2 + 1
        }
    }
    gen_config_yaml(
        args.output_root,
        audio_root=args.output_root.as_posix(),
        specaugment_policy="lb",
        feature_transform=["utterance_cmvn", "delta_deltas"],
        extra=extra,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", required=True, type=Path, help="source audio directory"
    )
    parser.add_argument(
        "--target-dir", required=True, type=Path, help="target audio directory"
    )
    parser.add_argument(
        "--data-split",
        default=["train", "valid", "test"],
        nargs="+",
        help="data split names",
    )
    parser.add_argument(
        "--output-root", required=True, type=Path, help="output directory"
    )
    # target feature related
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--f-min", type=int, default=20)
    parser.add_argument("--f-max", type=int, default=8000)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--normalize-volume", "-n", action="store_true")

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
