# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
import sys

import torch
import torch.nn.functional as F

from .. import FairseqDataset


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = (
            min_sample_size if min_sample_size is not None else self.max_sample_size
        )
        self.min_length = min_length
        self.shuffle = shuffle

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        def resample(x, factor):
            return F.interpolate(x.view(1, 1, -1), scale_factor=factor).squeeze()

        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            factor = self.sample_rate / curr_sample_rate
            feats = resample(feats, factor)

        assert feats.dim() == 1, feats.dim()
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [
            s for s in samples if s["source"] is not None and len(s["source"]) > 0
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]
        target_size = min(min(sizes), self.max_sample_size)

        if target_size < self.min_length:
            return {}

        if self.min_sample_size < target_size:
            target_size = np.random.randint(self.min_sample_size, target_size + 1)

        collated_sources = sources[0].new(len(sources), target_size)
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            assert diff >= 0
            if diff == 0:
                collated_sources[i] = source
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {"source": collated_sources},
        }

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
        )

        self.fnames = []

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                self.fnames.append(items[0])
                self.sizes.append(int(items[1]))

    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {"id": index, "source": feats}
