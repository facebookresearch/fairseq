#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import glob
import os
from shutil import copy

import h5py
import numpy as np
import soundfile as sf
import torch
import tqdm
from fairseq.models.wav2vec.wav2vec import Wav2VecModel
from torch import nn


def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3


class PretrainedWav2VecModel(nn.Module):
    def __init__(self, fname):
        super().__init__()

        checkpoint = torch.load(fname)
        self.args = checkpoint["args"]
        model = Wav2VecModel.build_model(self.args, None)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self.model = model

    def forward(self, x):
        with torch.no_grad():
            z = self.model.feature_extractor(x)
            if isinstance(z, tuple):
                z = z[0]
            c = self.model.feature_aggregator(z)
        return z, c


class EmbeddingWriterConfig(argparse.ArgumentParser):
    def __init__(self):
        super().__init__("Pre-compute embeddings for wav2letter++ datasets")

        kwargs = {"action": "store", "type": str, "required": True}

        self.add_argument("--input", "-i", help="Input Directory", **kwargs)
        self.add_argument("--output", "-o", help="Output Directory", **kwargs)
        self.add_argument("--model", help="Path to model checkpoint", **kwargs)
        self.add_argument("--split", help="Dataset Splits", nargs="+", **kwargs)
        self.add_argument(
            "--ext", default="wav", required=False, help="Audio file extension"
        )

        self.add_argument(
            "--no-copy-labels",
            action="store_true",
            help="Do not copy label files. Useful for large datasets, use --targetdir in wav2letter then.",
        )
        self.add_argument(
            "--use-feat",
            action="store_true",
            help="Use the feature vector ('z') instead of context vector ('c') for features",
        )
        self.add_argument("--gpu", help="GPU to use", default=0, type=int)


class Prediction:
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            z, c = self.model(x.unsqueeze(0))

        return z.squeeze(0).cpu().numpy(), c.squeeze(0).cpu().numpy()


class H5Writer:
    """ Write features as hdf5 file in wav2letter++ compatible format """

    def __init__(self, fname):
        self.fname = fname
        os.makedirs(os.path.dirname(self.fname), exist_ok=True)

    def write(self, data):
        channel, T = data.shape

        with h5py.File(self.fname, "w") as out_ds:
            data = data.T.flatten()
            out_ds["features"] = data
            out_ds["info"] = np.array([16e3 // 160, T, channel])


class EmbeddingDatasetWriter(object):
    """Given a model and a wav2letter++ dataset, pre-compute and store embeddings

    Args:
        input_root, str :
            Path to the wav2letter++ dataset
        output_root, str :
            Desired output directory. Will be created if non-existent
        split, str :
            Dataset split
    """

    def __init__(
        self,
        input_root,
        output_root,
        split,
        model_fname,
        extension="wav",
        gpu=0,
        verbose=False,
        use_feat=False,
    ):

        assert os.path.exists(model_fname)

        self.model_fname = model_fname
        self.model = Prediction(self.model_fname, gpu)

        self.input_root = input_root
        self.output_root = output_root
        self.split = split
        self.verbose = verbose
        self.extension = extension
        self.use_feat = use_feat

        assert os.path.exists(self.input_path), "Input path '{}' does not exist".format(
            self.input_path
        )

    def _progress(self, iterable, **kwargs):
        if self.verbose:
            return tqdm.tqdm(iterable, **kwargs)
        return iterable

    def require_output_path(self, fname=None):
        path = self.get_output_path(fname)
        os.makedirs(path, exist_ok=True)

    @property
    def input_path(self):
        return self.get_input_path()

    @property
    def output_path(self):
        return self.get_output_path()

    def get_input_path(self, fname=None):
        if fname is None:
            return os.path.join(self.input_root, self.split)
        return os.path.join(self.get_input_path(), fname)

    def get_output_path(self, fname=None):
        if fname is None:
            return os.path.join(self.output_root, self.split)
        return os.path.join(self.get_output_path(), fname)

    def copy_labels(self):
        self.require_output_path()

        labels = list(
            filter(
                lambda x: self.extension not in x, glob.glob(self.get_input_path("*"))
            )
        )
        for fname in tqdm.tqdm(labels):
            copy(fname, self.output_path)

    @property
    def input_fnames(self):
        return sorted(glob.glob(self.get_input_path("*.{}".format(self.extension))))

    def __len__(self):
        return len(self.input_fnames)

    def write_features(self):

        paths = self.input_fnames

        fnames_context = map(
            lambda x: os.path.join(
                self.output_path, x.replace("." + self.extension, ".h5context")
            ),
            map(os.path.basename, paths),
        )

        for name, target_fname in self._progress(
            zip(paths, fnames_context), total=len(self)
        ):
            wav, sr = read_audio(name)
            z, c = self.model(wav)
            feat = z if self.use_feat else c
            writer = H5Writer(target_fname)
            writer.write(feat)

    def __repr__(self):

        return "EmbeddingDatasetWriter ({n_files} files)\n\tinput:\t{input_root}\n\toutput:\t{output_root}\n\tsplit:\t{split})".format(
            n_files=len(self), **self.__dict__
        )


if __name__ == "__main__":

    args = EmbeddingWriterConfig().parse_args()

    for split in args.split:

        writer = EmbeddingDatasetWriter(
            input_root=args.input,
            output_root=args.output,
            split=split,
            model_fname=args.model,
            gpu=args.gpu,
            extension=args.ext,
            use_feat=args.use_feat,
        )

        print(writer)
        writer.require_output_path()

        print("Writing Features...")
        writer.write_features()
        print("Done.")

        if not args.no_copy_labels:
            print("Copying label data...")
            writer.copy_labels()
            print("Done.")
