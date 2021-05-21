#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import math
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile

from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="mean pools representations by compressing uniform splits of the data"
    )
    # fmt: off
    parser.add_argument('source', help='directory with features')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--subsample-rate', type=float, default=0.5, help='size to subsample data to')

    parser.add_argument('--remove-extra', action='store_true', help='if true, removes extra states that cant be pooled, otherwise pads with 0s')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    source_path = osp.join(args.source, args.split)

    print(f"data path: {source_path}")

    features = np.load(source_path + ".npy", mmap_mode="r")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = osp.join(args.save_dir, args.split)

    copyfile(source_path + ".tsv", save_path + ".tsv")
    copyfile(source_path + ".phn", save_path + ".phn")
    copyfile(source_path + ".wrd", save_path + ".wrd")

    if osp.exists(save_path + ".npy"):
        os.remove(save_path + ".npy")
    npaa = NpyAppendArray(save_path + ".npy")

    with open(source_path + ".lengths", "r") as lf:
        lengths = lf.readlines()

    fsz = features.shape[-1]
    start = 0
    with torch.no_grad():
        with open(save_path + ".lengths", "w") as lengths_out:
            for length in tqdm.tqdm(lengths):
                length = int(length)
                end = start + length
                feats = features[start:end]
                start += length
                x = torch.from_numpy(feats).cuda()
                target_num = math.ceil(length * args.subsample_rate)
                rem = length % target_num

                if rem > 0:
                    if args.remove_extra:
                        to_rem = target_num - rem
                        target_num -= 1
                        x = x[:-to_rem]
                    else:
                        to_add = target_num - rem
                        x = F.pad(x, [0, 0, 0, to_add])
                        x[-to_add:] = x[-to_add - 1]

                x = x.view(target_num, -1, fsz)
                x = x.mean(dim=-2)
                print(target_num, file=lengths_out)
                npaa.append(x.cpu().numpy())


if __name__ == "__main__":
    main()
