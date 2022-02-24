# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F

from . import FairseqDataset
from .data_utils import get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

import h5py
import cv2 as cv
import numpy as np
from .vision_transform import ToTensor, Normalize, RandomCrop, CenterCrop, RandomHorizontalFlip
from torchvision import transforms

logger = logging.getLogger(__name__)


class RawVideoDataset(FairseqDataset):
    def __init__(
        self,
        split,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
    ):
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.sizes = 0
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return self.sizes

    def mark(self):
        """
        return True as a special mark to set the batch_sampler
        """
        return True
    def postprocess(self, vidInp):
        # FIXME: Mean and Variance
        # assert feats.dim() == 3, feats.dim()
        if self.split == 'train':
            transform = transforms.Compose([
                ToTensor(),
                RandomCrop(112),
                RandomHorizontalFlip(0.5),
                Normalize(mean=[0.421], std=[0.165])
            ])
        else:
            transform = transforms.Compose([
                ToTensor(),
                CenterCrop(112),
                Normalize(mean=[0.421], std=[0.165])
            ])
        vidInp = torch.tensor(vidInp).unsqueeze(1)
        feats = transform(vidInp)

        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

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

        collated_sources = sources[0].new_zeros(len(sources), target_size, sources[0].shape[1], sources[0].shape[2], sources[0].shape[2])
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
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)


        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

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
            order = np.random.permutation(len(self))
            return order
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )


class FileVideoDataset(RawVideoDataset):
    def __init__(
        self,
        split,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__(
            split=split,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
        )
        self.text_compressor = TextCompressor(level=text_compression_level)
        self.dataset = h5py.File(manifest_path, "r")['png']
        self.sizes = len(self.dataset)

        # skipped = 0
        # self.fnames = []
        # sizes = []
        # self.skipped_indices = set()
        #
        # with open(manifest_path, "r") as f:
        #     self.root_dir = f.readline().strip()
        #     for i, line in enumerate(f):
        #         items = line.strip().split("\t")
        #         assert len(items) == 2, line
        #         sz = int(items[1])
        #         if min_sample_size is not None and sz < min_sample_size:
        #             skipped += 1
        #             self.skipped_indices.add(i)
        #             continue
        #         self.fnames.append(self.text_compressor.compress(items[0]))
        #         sizes.append(sz)
        # logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")
        #
        # self.sizes = np.array(sizes, dtype=np.int64)

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        # FIXME: postprocess
        vidInp = cv.imdecode(self.dataset[index], cv.IMREAD_COLOR)
        vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
        feats = self.postprocess(vidInp)
        return {"id": index, "source": feats}
