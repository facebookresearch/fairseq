# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from the BeiT implementation which can be found here:
# https://github.com/microsoft/unilm/tree/master/beit

import logging

from dataclasses import dataclass
from typing import Any

from omegaconf import II, MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model


logger = logging.getLogger(__name__)


@dataclass
class Data2VecImageClassificationConfig(FairseqDataclass):
    model_path: str = MISSING
    no_pretrained_weights: bool = False
    num_classes: int = 1000
    mixup: float = 0.8
    cutmix: float = 1.0
    label_smoothing: float = 0.1

    pretrained_model_args: Any = None
    data: str = II("task.data")


@register_model(
    "data2vec_image_classification", dataclass=Data2VecImageClassificationConfig
)
class Data2VecImageClassificationModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecImageClassificationConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.pretrained_model_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.model_path, {})
            pretrained_args = state.get("cfg", None)
            pretrained_args.criterion = None
            pretrained_args.lr_scheduler = None
            cfg.pretrained_model_args = pretrained_args

            logger.info(pretrained_args)
        else:
            state = None
            pretrained_args = cfg.pretrained_model_args

        pretrained_args.task.data = cfg.data
        task = tasks.setup_task(pretrained_args.task)
        model = task.build_model(pretrained_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        self.model = model

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        self.fc_norm = nn.LayerNorm(pretrained_args.model.embed_dim)
        self.head = nn.Linear(pretrained_args.model.embed_dim, cfg.num_classes)

        self.head.weight.data.mul_(1e-3)
        self.head.bias.data.mul_(1e-3)

        self.mixup_fn = None

        if cfg.mixup > 0 or cfg.cutmix > 0:
            from timm.data import Mixup

            self.mixup_fn = Mixup(
                mixup_alpha=cfg.mixup,
                cutmix_alpha=cfg.cutmix,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=cfg.label_smoothing,
                num_classes=cfg.num_classes,
            )

    def load_model_weights(self, state, model, cfg):
        if "_ema" in state["model"]:
            del state["model"]["_ema"]
        model.load_state_dict(state["model"], strict=True)

    @classmethod
    def build_model(cls, cfg: Data2VecImageClassificationConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def forward(
        self,
        img,
        label=None,
    ):
        if self.training and self.mixup_fn is not None and label is not None:
            img, label = self.mixup_fn(img, label)

        x = self.model(img, mask=False)
        x = x[:, 1:]
        x = self.fc_norm(x.mean(1))
        x = self.head(x)

        if label is None:
            return x

        if self.training and self.mixup_fn is not None:
            loss = -label * F.log_softmax(x.float(), dim=-1)
        else:
            loss = F.cross_entropy(
                x.float(),
                label,
                label_smoothing=self.cfg.label_smoothing if self.training else 0,
                reduction="none",
            )

        result = {
            "losses": {"regression": loss},
            "sample_size": img.size(0),
        }

        if not self.training:
            with torch.no_grad():
                pred = x.argmax(-1)
                correct = (pred == label).sum()
                result["correct"] = correct

        return result
