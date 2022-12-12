# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys
import os.path as osp

from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING

import torch
from torchvision import transforms

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

try:
    from ..data import ImageDataset
except:
    sys.path.append("..")
    from data import ImageDataset

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class ImagePretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    input_size: int = 224
    normalization_mean: List[float] = (0.485, 0.456, 0.406)
    normalization_std: List[float] = (0.229, 0.224, 0.225)


@register_task("image_pretraining", dataclass=ImagePretrainingConfig)
class ImagePretrainingTask(FairseqTask):
    """ """

    cfg: ImagePretrainingConfig

    @classmethod
    def setup_task(cls, cfg: ImagePretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        cfg = task_cfg or self.cfg

        path_with_split = osp.join(data_path, split)
        if osp.exists(path_with_split):
            data_path = path_with_split

        transform = transforms.Compose(
            [
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(
                    size=cfg.input_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(cfg.normalization_mean),
                    std=torch.tensor(cfg.normalization_std),
                ),
            ]
        )

        logger.info(transform)

        self.datasets[split] = ImageDataset(
            root=data_path,
            extensions=IMG_EXTENSIONS,
            load_classes=False,
            transform=transform,
        )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize
