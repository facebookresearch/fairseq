# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os.path as osp
import logging

from dataclasses import dataclass
import torch
from torchvision import transforms

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.logging import metrics

try:
    from ..data import ImageDataset
except:
    import sys

    sys.path.append("..")
    from data import ImageDataset

from .image_pretraining import (
    ImagePretrainingConfig,
    ImagePretrainingTask,
    IMG_EXTENSIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageClassificationConfig(ImagePretrainingConfig):
    pass


@register_task("image_classification", dataclass=ImageClassificationConfig)
class ImageClassificationTask(ImagePretrainingTask):

    cfg: ImageClassificationConfig

    @classmethod
    def setup_task(cls, cfg: ImageClassificationConfig, **kwargs):
        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        cfg = task_cfg or self.cfg

        path_with_split = osp.join(data_path, split)
        if osp.exists(path_with_split):
            data_path = path_with_split

        from timm.data import create_transform

        if split == "train":
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=cfg.input_size,
                is_training=True,
                auto_augment="rand-m9-mstd0.5-inc1",
                interpolation="bicubic",
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                mean=cfg.normalization_mean,
                std=cfg.normalization_std,
            )
            if not cfg.input_size > 32:
                transform.transforms[0] = transforms.RandomCrop(
                    cfg.input_size, padding=4
                )
        else:
            t = []
            if cfg.input_size > 32:
                crop_pct = 1
                if cfg.input_size < 384:
                    crop_pct = 224 / 256
                size = int(cfg.input_size / crop_pct)
                t.append(
                    transforms.Resize(
                        size, interpolation=3
                    ),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(cfg.input_size))

            t.append(transforms.ToTensor())
            t.append(
                transforms.Normalize(cfg.normalization_mean, cfg.normalization_std)
            )
            transform = transforms.Compose(t)
            logger.info(transform)

        self.datasets[split] = ImageDataset(
            root=data_path,
            extensions=IMG_EXTENSIONS,
            load_classes=True,
            transform=transform,
        )
        for k in self.datasets.keys():
            if k != split:
                assert self.datasets[k].classes == self.datasets[split].classes

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            if hasattr(actualized_cfg, "pretrained_model_args"):
                model_cfg.pretrained_model_args = actualized_cfg.pretrained_model_args

        return model

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if "correct" in logging_outputs[0]:
            zero = torch.scalar_tensor(0.0)
            correct = sum(log.get("correct", zero) for log in logging_outputs)
            metrics.log_scalar_sum("_correct", correct)

            metrics.log_derived(
                "accuracy",
                lambda meters: 100 * meters["_correct"].sum / meters["sample_size"].sum,
            )
