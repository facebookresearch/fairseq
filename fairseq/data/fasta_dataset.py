# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch


def fasta_file_path(prefix_path):
    return prefix_path + ".fasta"


class FastaDataset(torch.utils.data.Dataset):
    """
    For loading protein sequence datasets in the common FASTA data format
    """

    def __init__(self, path: str, cache_indices=False):
        self.fn = fasta_file_path(path)
        self.threadlocal = threading.local()
        self.cache = Path(f"{path}.fasta.idx.npy")
        if cache_indices:
            if self.cache.exists():
                self.offsets, self.sizes = np.load(self.cache)
            else:
                self.offsets, self.sizes = self._build_index(path)
                np.save(self.cache, np.stack([self.offsets, self.sizes]))
        else:
            self.offsets, self.sizes = self._build_index(path)

    def _get_file(self):
        if not hasattr(self.threadlocal, "f"):
            self.threadlocal.f = open(self.fn, "r")
        return self.threadlocal.f

    def __getitem__(self, idx):
        f = self._get_file()
        f.seek(self.offsets[idx])
        desc = f.readline().strip()
        line = f.readline()
        seq = ""
        while line != "" and line[0] != ">":
            seq += line.strip()
            line = f.readline()
        return desc, seq

    def __len__(self):
        return self.offsets.size

    def _build_index(self, path: str):
        # Use grep and awk to get 100M/s on local SSD.
        # Should process your enormous 100G fasta in ~10 min single core...
        path = fasta_file_path(path)
        bytes_offsets = subprocess.check_output(
            f"cat {path} | tqdm --bytes --total $(wc -c < {path})"
            "| grep --byte-offset '^>' -o | cut -d: -f1",
            shell=True,
        )
        fasta_lengths = subprocess.check_output(
            f"cat {path} | tqdm --bytes --total $(wc -c < {path})"
            "| awk '/^>/ {print \"\";next;} { printf(\"%s\",$0);}' | tail -n+2 | awk '{print length($1)}'",
            shell=True,
        )
        bytes_np = np.fromstring(bytes_offsets, dtype=np.int64, sep=" ")
        sizes_np = np.fromstring(fasta_lengths, dtype=np.int64, sep=" ")
        return bytes_np, sizes_np

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "f"):
            self.threadlocal.f.close()
            del self.threadlocal.f

    @staticmethod
    def exists(path):
        return os.path.exists(fasta_file_path(path))


class EncodedFastaDataset(FastaDataset):
    """
    The FastaDataset returns raw sequences - this allows us to return
    indices with a dictionary instead.
    """

    def __init__(self, path, dictionary):
        super().__init__(path, cache_indices=True)
        self.dictionary = dictionary

    def __getitem__(self, idx):
        desc, seq = super().__getitem__(idx)
        return self.dictionary.encode_line(seq, line_tokenizer=list).long()
