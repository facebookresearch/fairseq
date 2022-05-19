#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import soundfile as sf
from examples.speech_to_text.prep_mustc_data import (
    MUSTC
)

from tqdm import tqdm

log = logging.getLogger(__name__)


def main(args):
    root = Path(args.data_root).absolute()
    lang = args.lang
    split = args.split

    cur_root = root / f"en-{lang}"
    assert cur_root.is_dir(), (
        f"{cur_root.as_posix()} does not exist. Skipped."
    )

    dataset = MUSTC(root.as_posix(), lang, split)
    output = Path(args.output).absolute()
    output.mkdir(exist_ok=True)
    f_text = open(output / f"{split}.{lang}", "w")
    f_wav_list = open(output / f"{split}.wav_list", "w")
    for waveform, sample_rate, _, text, _, utt_id in tqdm(dataset):
        sf.write(
            output / f"{utt_id}.wav",
            waveform.squeeze(0).numpy(),
            samplerate=int(sample_rate)
        )
        f_text.write(text + "\n")
        f_wav_list.write(str(output / f"{utt_id}.wav") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--task", required=True, type=str, choices=["asr", "st"])
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", required=True, choices=MUSTC.SPLITS)
    args = parser.parse_args()

    main(args)
