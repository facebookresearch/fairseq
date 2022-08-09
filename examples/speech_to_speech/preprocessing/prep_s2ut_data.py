#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path

import soundfile as sf
from tqdm import tqdm
import pandas as pd

from examples.speech_to_speech.preprocessing.data_utils import (
    gen_config_yaml,
    load_units,
    process_units,
)
from examples.speech_to_text.data_utils import save_df_to_tsv

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames"]


def process(args):
    args.output_root.mkdir(exist_ok=True)

    print("Generating manifest...")
    for split in args.data_split:
        print(f"Processing {split}")

        # load target units
        target_unit_data = load_units(args.target_dir / f"{split}.txt")

        manifest = {c: [] for c in MANIFEST_COLUMNS}
        missing_tgt_audios = []
        src_audios = list(args.source_dir.glob(f"{split}/*.wav"))
        for src_audio in tqdm(src_audios):
            sample_id = src_audio.stem

            if sample_id not in target_unit_data:
                missing_tgt_audios.append(sample_id)
                continue

            src_n_frames = sf.info(src_audio.as_posix()).frames
            manifest["id"].append(sample_id)
            manifest["src_audio"].append(src_audio.as_posix())
            manifest["src_n_frames"].append(
                src_n_frames // 160
            )  # estimation of 10-ms frame for 16kHz audio

            target_units = process_units(target_unit_data[sample_id], args.reduce_unit)
            manifest["tgt_audio"].append(" ".join(target_units))
            manifest["tgt_n_frames"].append(len(target_units))

        print(f"Processed {len(manifest['id'])} samples")
        if len(missing_tgt_audios) > 0:
            print(
                f"{len(missing_tgt_audios)} with missing target data (first 3 examples: {', '.join(missing_tgt_audios[:3])})"
            )

        out_manifest = args.output_root / f"{split}.tsv"
        print(f"Writing manifest to {out_manifest}...")
        save_df_to_tsv(pd.DataFrame.from_dict(manifest), out_manifest)

    # Generate config YAML
    gen_config_yaml(
        args.output_root,
        specaugment_policy="lb",
        feature_transform=["utterance_cmvn"],
        vocoder_type="code_hifigan",
        vocoder_checkpoint=args.vocoder_checkpoint,
        vocoder_cfg=args.vocoder_cfg,
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
    parser.add_argument(
        "--reduce-unit",
        action="store_true",
        help="reduce a target unit sequence to a unique unit sequence, i.e. '1 1 1 2 2' -> '1 2'",
    )
    parser.add_argument(
        "--vocoder-checkpoint", default=None, type=str, help="vocoder checkpoint"
    )
    parser.add_argument(
        "--vocoder-cfg", default=None, type=str, help="vocoder config file"
    )

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
