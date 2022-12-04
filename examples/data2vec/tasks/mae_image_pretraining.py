# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys

from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING, II

from fairseq.data import SubsampleDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

try:
    from ..data import MaeImageDataset
except:
    sys.path.append("..")
    from data import MaeImageDataset

logger = logging.getLogger(__name__)


@dataclass
class ImageMaskingConfig:
    patch_size: int = II("model.modalities.image.patch_size")
    mask_prob: float = II("model.modalities.image.mask_prob")
    mask_prob_adjust: float = II("model.modalities.image.mask_prob_adjust")
    mask_length: int = II("model.modalities.image.mask_length")
    inverse_mask: bool = II("model.modalities.image.inverse_mask")
    mask_dropout: float = II("model.modalities.image.mask_dropout")
    clone_batch: int = II("model.clone_batch")
    expand_adjacent: bool = False
    non_overlapping: bool = False


@dataclass
class MaeImagePretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    multi_data: Optional[List[str]] = None
    input_size: int = 224
    local_cache_path: Optional[str] = None
    key: str = "imgs"

    beit_transforms: bool = False
    target_transform: bool = False
    no_transform: bool = False

    rebuild_batches: bool = True

    precompute_mask_config: Optional[ImageMaskingConfig] = None

    subsample: float = 1
    seed: int = II("common.seed")
    dataset_type: str = "imagefolder"


@register_task("mae_image_pretraining", dataclass=MaeImagePretrainingConfig)
class MaeImagePretrainingTask(FairseqTask):
    """ """

    cfg: MaeImagePretrainingConfig

    @classmethod
    def setup_task(cls, cfg: MaeImagePretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        cfg = task_cfg or self.cfg

        compute_mask = cfg.precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = cfg.precompute_mask_config

        self.datasets[split] = MaeImageDataset(
            root=data_path if cfg.multi_data is None else cfg.multi_data,
            split=split,
            input_size=cfg.input_size,
            local_cache_path=cfg.local_cache_path,
            key=cfg.key,
            beit_transforms=cfg.beit_transforms,
            target_transform=cfg.target_transform,
            no_transform=cfg.no_transform,
            compute_mask=compute_mask,
            dataset_type=cfg.dataset_type,
            **mask_args,
        )

        if cfg.subsample < 1:
            self.datasets[split] = SubsampleDataset(
                self.datasets[split],
                cfg.subsample,
                shuffle=True,
                seed=cfg.seed,
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
