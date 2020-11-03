# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
try:
    import soundfile as sf
except ImportError as err:
    print(err)
    print('try: pip install soundfile')

from .. import FairseqDataset


logger = logging.getLogger(__name__)
DBFS_COEF = 10.0 / torch.log10(torch.scalar_tensor(10.0))


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        noise_dir=None,
        noise_min_snr_db=3,
        noise_max_snr_db=40,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

        if noise_dir:
            assert os.path.exists(noise_dir)
        self.noise_dir = os.path.abspath(noise_dir) if noise_dir else None
        assert noise_min_snr_db != noise_max_snr_db
        assert noise_min_snr_db > 0
        if noise_min_snr_db > noise_max_snr_db:
            noise_min_snr_db, noise_max_snr_db = noise_max_snr_db, noise_min_snr_db
        self.noise_min_snr_db = noise_min_snr_db
        self.noise_max_snr_db = noise_max_snr_db
        self.noise_snr_range = noise_max_snr_db - noise_min_snr_db

        self.noise_files = []
        self.noise_files_len = 0

        if self.noise_dir:
            for dirpath, dirnames, filenames in os.walk(self.noise_dir):
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    self.noise_files.append(path)
            self.noise_files_len = len(self.noise_files)

        # self.noise_generator = gen() if self.noise_dir else None

    def sample_noise(self) -> torch.Tensor:
        try:
            path = self.noise_files[np.random.randint(self.noise_files_len)]
            wav, curr_sample_rate = sf.read(path)
            assert curr_sample_rate == self.sample_rate
            return torch.from_numpy(wav).float()
        except Exception as err:
            print(err)
            return self.sample_noise()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def wav_to_dbfs(self, feats: torch.Tensor) -> torch.Tensor:
        return DBFS_COEF * torch.log10(torch.dot(feats, feats) / len(feats) + 1e-8)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.noise_files_len:            
            noise: torch.Tensor = self.sample_noise()

            # peak protection
            max_gain_ratio = 1.0 / torch.max(noise)
            
            noise = noise.repeat(np.math.ceil(len(feats) / len(noise)))
            delta_len = len(noise) - len(feats)

            start = np.random.randint(delta_len) if delta_len > 0 else 0
            end = start + len(feats)

            seg_noise = noise[start:end]

            feats_dbfs = self.wav_to_dbfs(feats)
            seg_dbfs = self.wav_to_dbfs(seg_noise)
            current_snr_db = feats_dbfs - seg_dbfs
            target_snr_db = torch.rand(1) * self.noise_snr_range + self.noise_min_snr_db
            gain_db = current_snr_db - target_snr_db
            gain_ratio = torch.pow(10, gain_db / 20)

            feats += seg_noise * min(gain_ratio, max_gain_ratio)

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
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
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        noise_dir=None,
        noise_min_snr_db=3,
        noise_max_snr_db=40,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
            noise_dir=noise_dir,
            noise_min_snr_db=noise_min_snr_db,
            noise_max_snr_db=noise_max_snr_db,
        )

        self.fnames = []

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {"id": index, "source": feats}
