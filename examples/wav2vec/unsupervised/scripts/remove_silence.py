#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
get intervals from .vads file, specify output data, and this script removes silences and saves the audio data in out path folder
paths=shards/train.tsv
vads=shards/train.vads
python remove_silence.py --paths $paths --vads $vads
"""

import os
import argparse
import torch
import torchaudio
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--tsv", default="", type=str)
parser.add_argument("--vads", default="", type=str)
parser.add_argument("--out", type=str)
params = parser.parse_args()

# load paths
paths = []
with open(params.tsv) as f:
    root = next(f).rstrip()
    for line in f:
        paths.append(os.path.join(root, line.rstrip().split("\t")[0]))

# load vads
list_intervals = []
with open(params.vads) as f:
    for line in f:
        interval = [
            [int(w.split(":")[0]), int(w.split(":")[1])] for w in line.rstrip().split()
        ]
        list_intervals.append(interval)


# load audio and keep only intervals (i.e. remove silences)
for i in tqdm.trange(len(paths)):
    data, _ = torchaudio.load(paths[i])
    if len(list_intervals[i]) > 0:
        data_filtered = torch.cat(
            [data[0][int(it[0]) : int(it[1])] for it in list_intervals[i]]
        ).unsqueeze(0)
    else:
        data_filtered = data

    # YOU MAY NEED TO MODIFY THIS TO GET THE RIGHT SUBPATH
    # outpath = params.out + '/'.join(paths[i].split('/')[-1])
    outpath = params.out + "/" + "/".join(paths[i].split("/")[-2:])

    if not os.path.isdir("/".join(outpath.split("/")[:-1])):
        os.makedirs("/".join(outpath.split("/")[:-1]))
    if not os.path.exists(outpath):
        torchaudio.save(outpath, data_filtered, sample_rate=16000)
    else:
        print(outpath, "exists!")
