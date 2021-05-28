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
from shutil import copyfile

from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="transforms features via a given pca and stored them in target dir"
    )
    # fmt: off
    parser.add_argument('source', help='directory with features')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--pca-path', type=str, help='pca location. will append _A.npy and _b.npy', required=True)
    parser.add_argument('--batch-size', type=int, default=2048000, help='batch size')
    parser.add_argument('--unfiltered', action='store_true', help='process the unfiltered version')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    source_path = osp.join(args.source, args.split)
    data_poth = source_path + "_unfiltered" if args.unfiltered else source_path

    print(f"data path: {data_poth}")

    features = np.load(data_poth + ".npy", mmap_mode="r")
    pca_A = torch.from_numpy(np.load(args.pca_path + "_A.npy")).cuda()
    pca_b = torch.from_numpy(np.load(args.pca_path + "_b.npy")).cuda()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = osp.join(args.save_dir, args.split)

    copyfile(source_path + ".tsv", save_path + ".tsv")
    copyfile(data_poth + ".lengths", save_path + ".lengths")
    copyfile(source_path + ".phn", save_path + ".phn")
    copyfile(source_path + ".wrd", save_path + ".wrd")

    if osp.exists(save_path + ".npy"):
        os.remove(save_path + ".npy")
    npaa = NpyAppendArray(save_path + ".npy")

    batches = math.ceil(features.shape[0] / args.batch_size)

    with torch.no_grad():
        for b in tqdm.trange(batches):
            start = b * args.batch_size
            end = start + args.batch_size
            x = torch.from_numpy(features[start:end]).cuda()
            x = torch.matmul(x, pca_A) + pca_b
            npaa.append(x.cpu().numpy())


if __name__ == "__main__":
    main()
