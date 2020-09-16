#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import csv
import logging
import numpy as np
import pickle as pkl
import random
from fairseq.data.audio.speech_to_text_dataset import get_dataset_from_tsv
from fairseq.data.audio.feature_fetcher import fetch_features
try:
    import submitit
except:
    submitit=None

log = logging.getLogger(__name__)

COL_ID, COL_AUDIO = 'id', 'audio'

def load_tsv(tsv_path, audio_root=""):
    samples = []
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f, delimiter='\t', quotechar=None, doublequote=False,
            lineterminator='\n', quoting=csv.QUOTE_NONE
        )
        samples = [os.path.join(audio_root, dict(e)[COL_AUDIO]) for e in reader]
        assert len(samples) > 0
    return samples


def cal_stats(samples):
    features = np.concatenate([ fetch_features(sa).astype('float64')  for sa in samples ], axis=0)

    square_sums = (features ** 2).sum(axis=0)
    mean = features.mean(axis=0)

    features = np.subtract(features, mean)
    var = square_sums / features.shape[0] - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    return {"mean": mean.astype('float32'), "std": std.astype('float32')}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-path", required=True)
    parser.add_argument("--audio-root", default="", help="base directory feature npy")
    parser.add_argument("--gcmvn-path", help="path to save gcmvn pickle")
    parser.add_argument("--max-num", default=150000, type=int, help="maximum number of sentences to use")
    args = parser.parse_args()
    samples = load_tsv(args.tsv_path, args.audio_root)
    if args.max_num < len(samples):
        random.shuffle(samples)
        samples = samples[:args.max_num]

    stats = cal_stats(samples)
    with open(args.gcmvn_path, 'wb') as f:
        pkl.dump(stats, f)




if __name__ == "__main__":
    main()
