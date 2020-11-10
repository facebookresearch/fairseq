#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a flashlight (previously called wav2letter++) dataset
"""

import argparse
import glob
import os
import os.path as osp
import pprint

import soundfile as sf
import torch
import fairseq
from torch import nn
from torch.utils.data import DataLoader


try:
    import tqdm
except:
    print("Install tqdm to use --log-format=tqdm")


class FilesDataset:
    def __init__(self, files, labels):
        self.files = files
        if labels and osp.exists(labels):
            with open(labels, "r") as lbl_f:
                self.labels = [line.rstrip() for line in lbl_f]
        else:
            self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]

        wav, sr = sf.read(fname)
        assert sr == 16000

        wav = torch.from_numpy(wav).float()
        lbls = None
        if self.labels:
            if isinstance(self.labels, str):
                lbl_file = osp.splitext(fname)[0] + "." + self.labels
                with open(lbl_file, "r") as lblf:
                    lbls = lblf.readline()
                    assert lbls is not None
            else:
                lbls = self.labels[index]
        return wav, lbls

    def collate(self, batch):
        return batch


class ArgTypes:
    @staticmethod
    def existing_path(arg):
        arg = str(arg)
        assert osp.exists(arg), f"File {arg} does not exist"
        return arg

    @staticmethod
    def mkdir(arg):
        arg = str(arg)
        os.makedirs(arg, exist_ok=True)
        return arg


class DatasetWriter:
    def __init__(self):

        self.args = self.load_config()
        pprint.pprint(self.args.__dict__)

        self.model = self.load_model()

    def __getattr__(self, attr):
        return getattr(self.args, attr)

    def read_manifest(self, fname):

        with open(fname, "r") as fp:
            lines = fp.read().split("\n")
            root = lines.pop(0).strip()
            fnames = [
                osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0
            ]

        return fnames

    def process_splits(self):

        if self.args.shard is not None or self.args.num_shards is not None:
            assert self.args.shard is not None and self.args.num_shards is not None

        for split in self.splits:
            print(split)

            if self.extension == "tsv":
                datadir = osp.join(self.data_dir, f"{split}.{self.extension}")
                print("Reading manifest file: ", datadir)
                files = self.read_manifest(datadir)
            else:
                datadir = osp.join(self.data_dir, split, f"**/*.{self.extension}")
                files = glob.glob(datadir, recursive=True)

            assert len(files) > 0

            if self.args.shard is not None:
                files = files[self.args.shard :: self.args.num_shards]

            lbls = []
            with open(self.data_file(split), "w") as srcf:
                for line, lbl in self.iterate(files):
                    print(line, file=srcf)
                    if self.args.labels:
                        lbls.append(lbl + "\n")

            if self.args.labels:
                assert all(a is not None for a in lbls)
                with open(self.lbl_file(split), "w") as lblf:
                    lblf.writelines(lbls)

    def iterate(self, files):

        data = self.load_data(files)
        for samples in tqdm.tqdm(data, total=len(files) // 32):

            for wav, lbl in samples:
                x = wav.unsqueeze(0).float().cuda()

                div = 1
                while x.size(-1) // div > self.args.max_size:
                    div += 1

                xs = x.chunk(div, dim=-1)

                result = []
                for x in xs:
                    torch.cuda.empty_cache()
                    x = self.model.feature_extractor(x)
                    if self.quantize_location == "encoder":
                        with torch.no_grad():
                            _, idx = self.model.vector_quantizer.forward_idx(x)
                            idx = idx.squeeze(0).cpu()
                    else:
                        with torch.no_grad():
                            z = self.model.feature_aggregator(x)
                            _, idx = self.model.vector_quantizer.forward_idx(z)
                            idx = idx.squeeze(0).cpu()
                    result.append(idx)

                idx = torch.cat(result, dim=0)
                yield " ".join("-".join(map(str, a.tolist())) for a in idx), lbl

    def lbl_file(self, name):
        shard_part = "" if self.args.shard is None else f".{self.args.shard}"
        return osp.join(self.output_dir, f"{name}.lbl{shard_part}")

    def data_file(self, name):
        shard_part = "" if self.args.shard is None else f".{self.args.shard}"
        return osp.join(self.output_dir, f"{name}.src{shard_part}")

    def var_file(self):
        return osp.join(self.output_dir, f"vars.pt")

    def load_config(self):

        parser = argparse.ArgumentParser("Vector Quantized wav2vec features")

        # Model Arguments
        parser.add_argument("--checkpoint", type=ArgTypes.existing_path, required=True)
        parser.add_argument("--data-parallel", action="store_true")

        # Output Arguments
        parser.add_argument("--output-dir", type=ArgTypes.mkdir, required=True)

        # Data Arguments
        parser.add_argument("--data-dir", type=ArgTypes.existing_path, required=True)
        parser.add_argument("--splits", type=str, nargs="+", required=True)
        parser.add_argument("--extension", type=str, required=True)
        parser.add_argument("--labels", type=str, required=False)

        parser.add_argument("--shard", type=int, default=None)
        parser.add_argument("--num-shards", type=int, default=None)
        parser.add_argument("--max-size", type=int, default=1300000)

        # Logger Arguments
        parser.add_argument(
            "--log-format", type=str, choices=["none", "simple", "tqdm"]
        )

        return parser.parse_args()

    def load_data(self, fnames):

        dataset = FilesDataset(fnames, self.args.labels)
        loader = DataLoader(
            dataset, batch_size=32, collate_fn=dataset.collate, num_workers=8
        )
        return loader

    def load_model(self):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.checkpoint])
        model = model[0]

        self.quantize_location = getattr(cfg.model, "vq", "encoder")

        model.eval().float()
        model.cuda()

        if self.data_parallel:
            model = nn.DataParallel(model)

        return model

    def __call__(self):

        self.process_splits()

        if hasattr(self.model.feature_extractor, "vars") and (
            self.args.shard is None or self.args.shard == 0
        ):
            vars = (
                self.model.feature_extractor.vars.view(
                    self.model.feature_extractor.banks,
                    self.model.feature_extractor.num_vars,
                    -1,
                )
                .cpu()
                .detach()
            )
            print("writing learned latent variable embeddings: ", vars.shape)
            torch.save(vars, self.var_file())


if __name__ == "__main__":
    write_data = DatasetWriter()

    write_data()
    print("Done.")
