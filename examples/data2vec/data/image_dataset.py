# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging

import numpy as np
import os
from typing import Optional, Callable, Set

import torch

from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor

from fairseq.data import FairseqDataset


logger = logging.getLogger(__name__)


class ImageDataset(FairseqDataset, VisionDataset):
    def __init__(
        self,
        root: str,
        extensions: Set[str],
        load_classes: bool,
        transform: Optional[Callable] = None,
        shuffle=True,
    ):
        FairseqDataset.__init__(self)
        VisionDataset.__init__(self, root=root, transform=transform)

        self.shuffle = shuffle
        self.tensor_transform = ToTensor()

        self.classes = None
        self.labels = None
        if load_classes:
            classes = [d.name for d in os.scandir(root) if d.is_dir()]
            classes.sort()
            self.classes = {cls_name: i for i, cls_name in enumerate(classes)}
            logger.info(f"loaded {len(self.classes)} classes")
            self.labels = []

        def walk_path(root_path):
            for root, _, fnames in sorted(os.walk(root_path, followlinks=True)):
                for fname in sorted(fnames):
                    fname_ext = os.path.splitext(fname)
                    if fname_ext[-1].lower() not in extensions:
                        continue

                    path = os.path.join(root, fname)
                    yield path

        logger.info(f"finding images in {root}")
        if self.classes is not None:
            self.files = []
            self.labels = []
            for c, i in self.classes.items():
                for f in walk_path(os.path.join(root, c)):
                    self.files.append(f)
                    self.labels.append(i)
        else:
            self.files = [f for f in walk_path(root)]

        logger.info(f"loaded {len(self.files)} examples")

    def __getitem__(self, index):
        from PIL import Image

        fpath = self.files[index]

        with open(fpath, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is None:
            img = self.tensor_transform(img)
        else:
            img = self.transform(img)
            assert torch.is_tensor(img)

        res = {"id": index, "img": img}

        if self.labels is not None:
            res["label"] = self.labels[index]

        return res

    def __len__(self):
        return len(self.files)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        collated_img = torch.stack([s["img"] for s in samples], dim=0)

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "img": collated_img,
            },
        }

        if "label" in samples[0]:
            res["net_input"]["label"] = torch.LongTensor([s["label"] for s in samples])

        return res

    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        return order[0]
