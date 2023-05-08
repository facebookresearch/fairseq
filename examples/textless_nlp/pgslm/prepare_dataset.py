# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing import Pool

import os
from collections import defaultdict
from itertools import starmap

import torch
from npy_append_array import NpyAppendArray
from tqdm import tqdm

from data_utils import dump_speaker_f0_stat, F0Stat, load_f0
from fairseq.data.codedataset import (
    ExpressiveCodeDataConfig,
    parse_manifest,
    F0_FRAME_SPACE,
    align_f0_to_durations,
)
from fairseq.tasks.speech_ulm_task import UnitDictionary


def load_meta(meta_path, split):
    config = ExpressiveCodeDataConfig(meta_path)
    manifest_path = config.manifests[split]
    dictionary = UnitDictionary(n_units=config.n_units)
    audio_paths, codes, durs, speakers = parse_manifest(manifest_path, dictionary)
    return config, audio_paths, codes, durs, speakers


def _align_f0(f0, dur, ratio, frm_tol=5):
    if f0 is None:
        seg_f0 = torch.zeros_like(dur, dtype=torch.float)
    else:
        seg_f0 = align_f0_to_durations(f0, dur, ratio, tol=frm_tol * ratio)
    return seg_f0.numpy()  # try a hacky stuff


def align_f0(path_to_f0, audio_paths, durs, ratio, mp=False):
    chunk_size = 2000
    num_procs = 40
    iterable = ((path_to_f0[p], d, ratio) for p, d in zip(audio_paths, durs))

    seg_f0s = []
    if mp:
        with Pool(num_procs) as pool:
            iterator = tqdm(
                pool.istarmap(_align_f0, iterable, chunk_size),
                desc="align f0",
                total=len(durs),
            )
            for seg_f0 in iterator:
                seg_f0s.append(torch.from_numpy(seg_f0).float())
    else:
        iterator = tqdm(starmap(_align_f0, iterable), desc="align f0", total=len(durs))
        for seg_f0 in iterator:
            seg_f0s.append(torch.from_numpy(seg_f0).float())

    return seg_f0s


def prepare_seg_data(config, audio_paths, codes, durs, speakers, path_to_f0):
    ratio = config.code_hop_size / (config.sampling_rate * F0_FRAME_SPACE)
    seg_f0s = align_f0(path_to_f0, audio_paths, durs, ratio)
    data = {
        "codes": codes,
        "duration": durs,
        "f0": seg_f0s,
        "speaker": speakers,
        "path": audio_paths,
    }
    return data


def dump_seg_data(data, out_prefix):
    key_targs = {
        "codes": f"{out_prefix}.code.npy",
        "duration": f"{out_prefix}.dur.npy",
        "f0": f"{out_prefix}.f0.npy",
    }
    for key, targ in key_targs.items():
        assert not os.path.exists(targ)
        npaa = NpyAppendArray(targ)
        for utt_data in tqdm(data[key], desc=f"dumping {key}"):
            npaa.append(utt_data.numpy())

    assert not os.path.exists(f"{out_prefix}.path.txt")
    with open(f"{out_prefix}.path.txt", "w") as f:
        for x in data["path"]:
            f.write(f"{str(x)}\n")

    assert not os.path.exists(f"{out_prefix}.leng.txt")
    with open(f"{out_prefix}.leng.txt", "w") as f:
        for x in data["codes"]:
            f.write(f"{len(x)}\n")

    assert not os.path.exists(f"{out_prefix}.speaker.txt")
    with open(f"{out_prefix}.speaker.txt", "w") as f:
        for x in data["speaker"]:
            f.write(f"{str(x)}\n")

    print(f"wrote to files with prefix {out_prefix}")


def main(meta_path, f0_dir, splits, nshards_list):
    speaker_to_stat = defaultdict(F0Stat)
    if len(nshards_list) == 1:
        nshards_list = nshards_list * len(splits)
    else:
        assert len(nshards_list) == len(splits)

    for split, nshards in zip(splits, nshards_list):
        config, audio_paths, codes, durs, speakers = load_meta(meta_path, split)
        path_to_f0 = load_f0(f"{f0_dir}/{split}", nshards)

        # segment-level data
        data = prepare_seg_data(config, audio_paths, codes, durs, speakers, path_to_f0)
        dump_seg_data(data, config.manifests[split])

        # speaker f0
        for audio_path, speaker in tqdm(zip(audio_paths, speakers)):
            f0 = path_to_f0[audio_path]
            speaker_to_stat[speaker].update(f0)
        dump_speaker_f0_stat(speaker_to_stat, config.manifests[split])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path")
    parser.add_argument("f0_dir", help="out_dir from preprocess_f0")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument(
        "--nshards_list", type=int, nargs="+", default=[20], help="number of f0 shards"
    )
    args = parser.parse_args()
    print(args)

    main(**vars(args))
