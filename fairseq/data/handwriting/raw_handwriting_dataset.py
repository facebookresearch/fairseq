# Copyright (c) UWr and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
import numpy as np
import sys

import torch
import torch.nn.functional as F

from . import scribblelens
from .. import FairseqDataset


logger = logging.getLogger(__name__)


class RawHandwritingDataset(FairseqDataset):
    def __init__(
        self,
        max_sample_size=None,
        min_sample_size=None,
        pad_to_multiples_of=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__()

        # We don't really have a sampling rate - out of audio (JCh)
        # self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad_to_multiples_of = pad_to_multiples_of
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        # TODO(jch): verify if this makes sense, prob not!
        # if feats.dim() == 2:
        #     feats = feats.mean(-1)

        # # Doesn't make sense - JCh
        # # if curr_sample_rate != self.sample_rate:
        # #     raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        # assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size_dim1):
        size = wav.shape[1] #len(wav)
        diff = size - target_size_dim1
        if diff <= 0:
            return wav

        if self.shuffle:
            start = np.random.randint(0, diff + 1)
        else:
            # Deterministically pick the middle part
            start = (diff + 1) //2
        end = size - diff + start
        return wav[:, start:end]
        
    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [s.shape[1] for s in sources]
        heigths = [s.shape[0] for s in sources]
        assert all([h==heigths[0] for h in heigths])

        pad_to_multiples_of = self.pad_to_multiples_of
        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
            if pad_to_multiples_of:
                # round up to pad_to_multiples_of
                target_size = ((target_size + pad_to_multiples_of - 1) // pad_to_multiples_of) * pad_to_multiples_of
        else:
            target_size = min(min(sizes), self.max_sample_size)
            if pad_to_multiples_of:
                # round down to pad_to_multiples_of
                target_size = (target_size // pad_to_multiples_of) * pad_to_multiples_of

        collated_sources = sources[0].new_zeros((len(sources), heigths[0], target_size))
        pad_shape = list(collated_sources.shape)
        pad_shape[1] = 1  # we mask all pixels in exactly the same way
        padding_mask = (
            torch.BoolTensor(size=pad_shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((heigths[0], -diff), 0.0)],
                    dim=1
                )
                padding_mask[i, :, diff:] = True
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


class FileHandwritingDataset(RawHandwritingDataset):
    def __init__(
        self,
        dist_root,
        split,
        max_sample_size=None,
        min_sample_size=None,
        pad_to_multiples_of=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            pad_to_multiples_of=pad_to_multiples_of,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )
        self.dataset = scribblelens.ScribbleLensDataset(
            root=dist_root + '/scribblelens.corpus.v1.2.zip',           # Path to zip with images
            alignment_root=dist_root + '/scribblelens.paths.1.4b.zip',  # Path to the alignments, i.e. info aou char boundaries
            slice='tasman',                                     # Part of the data, here a single scribe https://en.wikipedia.org/wiki/Abel_Tasman
            split=split,                                      # Train, test, valid or unsupervised. Train/Test/Valid have character transcripts, unspuervised has only images
            # Not used in the simple ScribbleLens loader
            transcript_mode=5,                                  # Legacy space handling, has to be like that
            vocabulary=dist_root + '/tasman.alphabet.plus.space.mode5.json',  # Path
        )

        for data in self.dataset:
            sizeHere = data['image'].shape
            #print(sizeHere)
            self.sizes.append(sizeHere[0])  # 1/2 dim TODO?

        # self.fnames = []

        # skipped = 0
        # with open(manifest_path, "r") as f:
        #     self.root_dir = f.readline().strip()
        #     for line in f:
        #         items = line.strip().split("\t")
        #         assert len(items) == 2, line
        #         sz = int(items[1])
        #         if min_length is not None and sz < min_length:
        #             skipped += 1
        #             continue
        #         self.fnames.append(items[0])
        #         self.sizes.append(sz)
        # logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")



    def __getitem__(self, index):
        # import soundfile as sf

        # fname = os.path.join(self.root_dir, self.fnames[index])
        # wav, curr_sample_rate = sf.read(fname)
        # feats = torch.from_numpy(wav).float()
        # feats = self.postprocess(feats, curr_sample_rate)
        feats = self.dataset[index]['image'][:,:,0]
        return {"id": index, "source": feats.T}