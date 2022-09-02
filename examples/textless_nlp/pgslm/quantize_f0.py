# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from data_utils import dump_speaker_f0_stat, F0Stat, load_audio_path, load_f0


def load_speaker(path):
    speakers = []
    with open(path) as f:
        for line in f.readlines():
            sample = eval(line.strip())
            assert "speaker" in sample
            speakers.append(sample["speaker"])
    return speakers


def quantize_f0(speaker_to_f0, f0_stats, nbins, normalize, log):
    f0_all = []
    for speaker, f0 in speaker_to_f0.items():
        f0 = f0.raw_data
        if log:
            f0 = f0.log()
        mean = f0_stats[speaker]["logf0_mean"] if log else f0_stats[speaker]["f0_mean"]
        std = f0_stats[speaker]["logf0_std"] if log else f0_stats[speaker]["f0_std"]
        if normalize == "mean":
            f0 = f0 - mean
        elif normalize == "meanstd":
            f0 = (f0 - mean) / std
        f0_all.extend(f0.tolist())

    hist, bin_x = np.histogram(f0_all, 100000)
    cum_hist = np.cumsum(hist) / len(f0_all) * 100

    f0_bin = {}
    for num_bin in nbins:
        bin_offset = []
        bin_size = 100 / num_bin
        threshold = bin_size
        for i in range(num_bin - 1):
            index = (np.abs(cum_hist - threshold)).argmin()
            bin_offset.append(bin_x[index])
            threshold += bin_size
        f0_bin[num_bin] = np.array(bin_offset)

    return f0_bin


def main(file_path, f0_dir, out_dir, out_prefix, nbins, nshards, normalize, log):
    audio_paths = load_audio_path(file_path)
    path_to_f0 = load_f0(f0_dir, nshards)

    speakers = load_speaker(file_path)
    speaker_to_f0 = defaultdict(partial(F0Stat, True))

    # speaker f0 stats
    for audio_path, speaker in tqdm(zip(audio_paths, speakers)):
        f0 = path_to_f0[audio_path]
        speaker_to_f0[speaker].update(f0)
    f0_stats = dump_speaker_f0_stat(speaker_to_f0, f"{out_dir}/{out_prefix}")

    # quantize
    f0_bin = quantize_f0(speaker_to_f0, f0_stats, nbins, normalize, log)
    log_suffix = "_log" if log else ""
    f0_bin_out_file = f"{out_dir}/{out_prefix}_{normalize}_norm{log_suffix}_f0_bin.th"
    torch.save(f0_bin, f0_bin_out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("f0_dir", help="out_dir from preprocess_f0")
    parser.add_argument("out_dir")
    parser.add_argument("out_prefix")
    parser.add_argument("--nbins", nargs="+", type=int, default=[32])
    parser.add_argument("--nshards", type=int, default=20, help="number of f0 shards")
    parser.add_argument(
        "--normalize", type=str, choices=["meanstd", "mean", "none"], default="mean"
    )
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    print(args)

    main(**vars(args))
