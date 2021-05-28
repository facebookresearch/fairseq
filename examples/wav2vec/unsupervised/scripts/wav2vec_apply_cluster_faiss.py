#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp
import numpy as np
import tqdm
import torch
import sys

import faiss
import torch.nn.functional as F

from wav2vec_cluster_faiss import parse_faiss_specs, Wav2VecFeatureReader


def get_parser():
    parser = argparse.ArgumentParser(description="apply clusters")
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='split to process', required=True)
    parser.add_argument('--labels', help='split to process', default="phn")
    parser.add_argument('--path', help='path to pca and centroids', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec model (if using wav2vec features)', required=True)
    parser.add_argument('--layer', '-l', type=int, help='which layer to read', default=14)
    parser.add_argument('--max-tsz', type=int, help='batch kmeans up to this much', default=14)
    # fmt: on

    return parser


def get_iterator(args):
    with open(osp.join(args.data, f"{args.split}.tsv"), "r") as fp, open(
        osp.join(args.data, f"{args.split}.{args.labels}"), "r"
    ) as lp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [line.rstrip() for line in lines if len(line) > 0]
        lbls = [line.rstrip() for line in lp]

        num = len(files)
        reader = Wav2VecFeatureReader(args.checkpoint, args.layer)

        def iterate():
            for fname, lbl in zip(files, lbls):
                file = osp.join(root, fname.split("\t")[0])
                feats = reader.get_feats(file)
                yield feats.data, fname, lbl

        return iterate, num, root


def main():
    parser = get_parser()
    args = parser.parse_args()

    spec = osp.basename(args.path)

    try:
        faiss_spec = parse_faiss_specs(spec.rstrip("/"))[0]
    except:
        print(spec)
        raise

    print("Faiss Spec:", faiss_spec, file=sys.stderr)

    if faiss_spec.pca:
        A = torch.from_numpy(np.load(osp.join(args.path, "pca_A.npy"))).cuda()
        b = torch.from_numpy(np.load(osp.join(args.path, "pca_b.npy"))).cuda()
        print("Loaded PCA", file=sys.stderr)

    centroids = np.load(osp.join(args.path, "centroids.npy"))
    print("Loaded centroids", centroids.shape, file=sys.stderr)

    res = faiss.StandardGpuResources()
    index_flat = (
        faiss.IndexFlatL2(centroids.shape[1])
        if not faiss_spec.sphere
        else faiss.IndexFlatIP(centroids.shape[1])
    )
    faiss_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    faiss_index.add(centroids)

    generator, num, root = get_iterator(args)
    iterator = generator()

    with torch.no_grad():
        with open(osp.join(args.path, f"{args.split}.src"), "w") as fp, open(
            osp.join(args.path, f"{args.split}.tsv"), "w"
        ) as pp, open(osp.join(args.path, f"{args.split}.{args.labels}"), "w") as lp:
            print(root, file=pp)
            for f, fname, lbl in tqdm.tqdm(iterator, total=num):
                if faiss_spec.pca:
                    f = torch.mm(f, A) + b
                if faiss_spec.norm:
                    f = F.normalize(f, p=2, dim=-1)

                f = f.cpu().numpy()

                _, z = faiss_index.search(f, 1)

                print(" ".join(str(x.item()) for x in z), file=fp)
                print(fname, file=pp)
                print(lbl, file=lp)


if __name__ == "__main__":
    main()
