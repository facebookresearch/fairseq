#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import os
import os.path as osp
import random
import numpy as np
import tqdm
import torch

from collections import namedtuple

import faiss

import fairseq
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec model (if using wav2vec features)', required=True)
    parser.add_argument('--sample-pct', '-r', type=float, help='percentage of timesteps to sample', default=0)
    parser.add_argument('--layer', '-l', type=int, help='which layer to read', default=14)
    parser.add_argument('--faiss-specs', '-f', type=str,
                        help='faiss index specs; separated by space '
                             'format is: PCAx_NORM_CLUSx_SPHERICAL -> '
                                'PCAx if exists first apply PCA '
                                'NORM if exists, normalize the vector by L2 norm '
                                'CLUSx must exist, cluster to x clusters '
                                'SPEHRICAL if exists, apply spherical kmeans',
                        default='l2')
    # fmt: on

    return parser


faiss_spec = namedtuple("faiss_spec", ["pca", "norm", "n_clus", "sphere", "spec_str"])


def parse_faiss_specs(specs_str):
    specs = []
    for ss in specs_str.split():
        comps = ss.split("_")
        pca = 0
        norm = False
        n_clus = 0
        sphere = False
        for c in comps:
            if c.startswith("PCA"):
                pca = int(c[3:])
            elif c == "NORM":
                norm = True
            elif c.startswith("CLUS"):
                n_clus = int(c[4:])
            elif c == "SPHERICAL":
                sphere = True
        assert n_clus > 0
        specs.append(
            faiss_spec(pca=pca, norm=norm, n_clus=n_clus, sphere=sphere, spec_str=ss)
        )
    return specs


class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer):
        state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(cp_file)

        self.layer = layer

        if "cfg" in state:
            w2v_args = state["cfg"]
            task = fairseq.tasks.setup_task(w2v_args.task)
            model = task.build_model(w2v_args.model)
        else:
            w2v_args = state["args"]
            task = fairseq.tasks.setup_task(w2v_args)
            model = task.build_model(w2v_args)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        model.cuda()
        self.model = model

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).view(1, -1).float().cuda()
            res = self.model(
                source=source, mask=False, features_only=True, layer=self.layer
            )
            return res["layer_results"][self.layer][0].squeeze(1)


def get_iterator(args):
    with open(args.data, "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]

        if getattr(args, "sample_pct", 0) > 0:
            files = random.sample(files, int(args.sample_pct * len(files)))
        num = len(files)
        reader = Wav2VecFeatureReader(args.checkpoint, args.layer)

        def iterate():
            for fname in files:
                feats = reader.get_feats(fname)
                yield feats.cpu().numpy()

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    faiss_specs = parse_faiss_specs(args.faiss_specs)
    print("Faiss Specs:", faiss_specs)

    feat_path = osp.join(args.save_dir, "features")
    if osp.exists(feat_path + ".npy"):
        feats = np.load(feat_path + ".npy")
    else:
        generator, num = get_iterator(args)
        iterator = generator()

        feats = []
        for f in tqdm.tqdm(iterator, total=num):
            feats.append(f)

        del iterator
        del generator

        feats = np.concatenate(feats)

        print(feats.shape)

        os.makedirs(args.save_dir, exist_ok=True)
        # np.save(feat_path, feats)

        gc.collect()
        torch.cuda.empty_cache()

    reload = False
    for spec in faiss_specs:
        print("Processing spec", spec)

        if reload:
            print("Reloading...")
            del feats
            gc.collect()
            feats = np.load(feat_path + ".npy")

        save_path = osp.join(args.save_dir, spec.spec_str)
        os.makedirs(save_path, exist_ok=True)
        d = feats.shape[-1]
        x = feats
        if spec.pca > 0:
            print("Computing PCA")
            pca = faiss.PCAMatrix(d, spec.pca)
            pca.train(x)
            d = spec.pca
            b = faiss.vector_to_array(pca.b)
            A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
            np.save(osp.join(save_path, "pca_A"), A.T)
            np.save(osp.join(save_path, "pca_b"), b)
            print("Applying PCA")
            x = pca.apply_py(x)

        if spec.norm:
            reload = spec.pca <= 0
            print("Normalizing")
            faiss.normalize_L2(x)

        print("Computing kmeans")
        kmeans = faiss.Kmeans(
            d,
            spec.n_clus,
            niter=50,
            verbose=True,
            spherical=spec.sphere,
            max_points_per_centroid=feats.shape[0],
            gpu=True,
            nredo=3,
        )
        kmeans.train(x)
        np.save(osp.join(save_path, "centroids"), kmeans.centroids)
        del kmeans
        del x
        gc.collect()


if __name__ == "__main__":
    main()
