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

    def __init__(self, manifest_path, sample_rate, max_sample_size=None, min_sample_size=None,
                 shuffle=True):
        super().__init__()

        self.sample_rate = sample_rate
        self.fnames = []
        self.sizes = []
        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.max_sample_size

        with open(manifest_path, 'r') as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split('\t')
                assert len(items) == 2, line
                self.fnames.append(items[0])
                self.sizes.append(int(items[1]))
        self.shuffle = shuffle

    def __getitem__(self, index):
        fname = os.path.join(self.root_dir, self.fnames[index])
        import soundfile as sf

        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()

        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            factor = self.sample_rate / curr_sample_rate
            feats = self.resample(feats, factor)

        assert feats.dim() == 1, feats.dim()

        return {
            'id': index,
            'source': feats,
        }

    def resample(self, x, factor):
        return F.interpolate(x.view(1, 1, -1), scale_factor=factor).squeeze()

    def __len__(self):
        return len(self.fnames)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        sources = [s['source'] for s in samples]
        sizes = [len(s) for s in sources]
        target_size = min(min(sizes), self.max_sample_size)

        if self.min_sample_size < target_size:
            target_size = np.random.randint(self.min_sample_size, target_size + 1)

        collated_sources = sources[0].new(len(sources), target_size)
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            assert diff >= 0
            if diff == 0:
                collated_sources[i] = source
            else:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
                collated_sources[i] = source[start:end]

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'net_input': {
                'source': collated_sources,
            },
        }

    def get_dummy_batch(
            self, num_tokens, max_positions, src_len=2048, tgt_len=128,
    ):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            src_len = min(src_len, max_positions)
        bsz = num_tokens // src_len
        return self.collater([
            {
                'id': i,
                'source': torch.rand(src_len),
            }
            for i in range(bsz)
        ])

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
