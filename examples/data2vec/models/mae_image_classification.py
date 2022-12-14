# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from the BeiT implementation which can be found here:
# https://github.com/microsoft/unilm/tree/master/beit

import logging

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
from omegaconf import II, MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks
from omegaconf import open_dict

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from .mae import interpolate_pos_embed


logger = logging.getLogger(__name__)


class PredictionMode(Enum):
    MEAN_POOLING = auto()
    CLS_TOKEN = auto()
    LIN_SOFTMAX = auto()


@dataclass
class MaeImageClassificationConfig(FairseqDataclass):
    model_path: str = MISSING
    no_pretrained_weights: bool = False
    linear_classifier: bool = False
    num_classes: int = 1000
    mixup: float = 0.8
    cutmix: float = 1.0
    label_smoothing: float = 0.1

    drop_path_rate: float = 0.1
    layer_decay: float = 0.65

    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"

    pretrained_model_args: Any = None
    data: str = II("task.data")

    norm_eps: Optional[float] = None

    remove_alibi: bool = False

    # regularization overwrites
    encoder_dropout: float = 0
    post_mlp_drop: float = 0
    attention_dropout: float = 0
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0

    prenet_layerdrop: float = 0
    prenet_dropout: float = 0

    use_fc_norm: bool = True
    prediction_mode: PredictionMode = PredictionMode.MEAN_POOLING

    no_decay_blocks: bool = True


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers


@register_model("mae_image_classification", dataclass=MaeImageClassificationConfig)
class MaeImageClassificationModel(BaseFairseqModel):
    def __init__(self, cfg: MaeImageClassificationConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.pretrained_model_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.model_path, {})
            pretrained_args = state.get("cfg", None)

            pretrained_args.criterion = None
            pretrained_args.lr_scheduler = None

            logger.info(pretrained_args.model)

            with open_dict(pretrained_args.model):
                pretrained_args.model.drop_path_rate = cfg.drop_path_rate
                if cfg.norm_eps is not None:
                    pretrained_args.model.norm_eps = cfg.norm_eps

            cfg.pretrained_model_args = pretrained_args

            logger.info(pretrained_args)
        else:
            state = None
            pretrained_args = cfg.pretrained_model_args

        if "data" in pretrained_args.task:
            pretrained_args.task.data = cfg.data
        elif "image" in pretrained_args.task:
            pretrained_args.task.image.data = cfg.data

        if "modalities" in pretrained_args.model:
            prenet_blocks = pretrained_args.model["modalities"]["image"]["prenet_depth"]
            model_blocks = pretrained_args.model["depth"]
            with open_dict(pretrained_args):
                dpr = np.linspace(0, cfg.drop_path_rate, model_blocks).tolist()
                pretrained_args.model["modalities"]["image"][
                    "start_drop_path_rate"
                ] = dpr[0]
                pretrained_args.model["modalities"]["image"][
                    "end_drop_path_rate"
                ] = max(0, dpr[prenet_blocks - 1])
                pretrained_args.model["start_drop_path_rate"] = dpr[prenet_blocks]
                pretrained_args.model["end_drop_path_rate"] = dpr[-1]

                if "mae_masking" in pretrained_args.model["modalities"]["image"]:
                    del pretrained_args.model["modalities"]["image"]["mae_masking"]

                if cfg.remove_alibi:
                    pretrained_args.model["modalities"]["image"][
                        "use_alibi_encoder"
                    ] = False
                    if (
                        state is not None
                        and "modality_encoders.IMAGE.alibi_bias" in state["model"]
                    ):
                        del state["model"]["modality_encoders.IMAGE.alibi_bias"]

                pretrained_args.model["encoder_dropout"] = cfg.encoder_dropout
                pretrained_args.model["post_mlp_drop"] = cfg.post_mlp_drop
                pretrained_args.model["attention_dropout"] = cfg.attention_dropout
                pretrained_args.model["activation_dropout"] = cfg.activation_dropout
                pretrained_args.model["dropout_input"] = cfg.dropout_input
                pretrained_args.model["layerdrop"] = cfg.layerdrop

                pretrained_args.model["modalities"]["image"][
                    "prenet_layerdrop"
                ] = cfg.prenet_layerdrop
                pretrained_args.model["modalities"]["image"][
                    "prenet_dropout"
                ] = cfg.prenet_dropout
        else:
            # not d2v multi
            with open_dict(pretrained_args):
                pretrained_args.model["drop_path_rate"] = cfg.drop_path_rate
                pretrained_args.model["block_dropout"] = cfg.encoder_dropout
                pretrained_args.model["attention_dropout"] = cfg.attention_dropout
                pretrained_args.model["activation_dropout"] = cfg.activation_dropout

        task = tasks.setup_task(pretrained_args.task)
        model = task.build_model(pretrained_args.model, from_checkpoint=True)

        self.d2v_multi = "data2vec_multi" in pretrained_args.model._name
        self.linear_classifier = cfg.linear_classifier

        self.model = model

        if state is not None and not cfg.no_pretrained_weights:
            interpolate_pos_embed(model, state)

            if "modality_encoders.IMAGE.positional_encoder.pos_embed" in state["model"]:
                state["model"][
                    "modality_encoders.IMAGE.positional_encoder.positions"
                ] = state["model"][
                    "modality_encoders.IMAGE.positional_encoder.pos_embed"
                ]
                del state["model"][
                    "modality_encoders.IMAGE.positional_encoder.pos_embed"
                ]
            if "modality_encoders.IMAGE.encoder_mask" in state["model"]:
                del state["model"]["modality_encoders.IMAGE.encoder_mask"]

            model.load_state_dict(state["model"], strict=True)

        if self.d2v_multi:
            model.remove_pretraining_modules(modality="image")
        else:
            model.remove_pretraining_modules()

        if self.linear_classifier:
            model.requires_grad_(False)

        self.fc_norm = None
        if self.cfg.use_fc_norm:
            self.fc_norm = nn.LayerNorm(pretrained_args.model.embed_dim, eps=1e-6)
            nn.init.constant_(self.fc_norm.bias, 0)
            nn.init.constant_(self.fc_norm.weight, 1.0)

        self.head = nn.Linear(pretrained_args.model.embed_dim, cfg.num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

        self.mixup_fn = None

        if cfg.mixup > 0 or cfg.cutmix > 0:
            from timm.data import Mixup

            self.mixup_fn = Mixup(
                mixup_alpha=cfg.mixup,
                cutmix_alpha=cfg.cutmix,
                cutmix_minmax=None,
                prob=cfg.mixup_prob,
                switch_prob=cfg.mixup_switch_prob,
                mode=cfg.mixup_mode,
                label_smoothing=cfg.label_smoothing,
                num_classes=cfg.num_classes,
            )

        if self.model.norm is not None:
            for pn, p in self.model.norm.named_parameters():
                if len(p.shape) == 1 or pn.endswith(".bias"):
                    p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        if self.fc_norm is not None:
            for pn, p in self.fc_norm.named_parameters():
                if len(p.shape) == 1 or pn.endswith(".bias"):
                    p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        for pn, p in self.head.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias"):
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        if self.d2v_multi:
            mod_encs = list(model.modality_encoders.values())
            assert len(mod_encs) == 1, len(mod_encs)
            blocks = list(mod_encs[0].context_encoder.blocks) + list(model.blocks)
        else:
            blocks = model.blocks

        num_layers = len(blocks) + 1
        layer_scales = list(
            cfg.layer_decay ** (num_layers - i) for i in range(num_layers + 1)
        )

        if self.d2v_multi:
            for n, p in self.model.named_parameters():
                optimizer_override_dict = {}

                if len(p.shape) == 1 or n.endswith(".bias"):
                    optimizer_override_dict["weight_decay_scale"] = 0

                p.optim_overrides = {"optimizer": optimizer_override_dict}

            if cfg.layer_decay > 0:
                for i, b in enumerate(blocks):
                    lid = i + 1
                    if layer_scales[lid] == 1.0:
                        continue

                    for n, p in b.named_parameters():
                        optim_override = getattr(p, "optim_overrides", {})
                        if "optimizer" not in optim_override:
                            optim_override["optimizer"] = {}

                        if cfg.no_decay_blocks:
                            optim_override["optimizer"]["lr_scale"] = layer_scales[lid]
                            p.optim_overrides = optim_override
                        else:
                            optim_override["optimizer"] = {
                                "lr_scale": layer_scales[lid]
                            }
                            p.optim_overrides = optim_override

        else:
            for n, p in self.model.named_parameters():
                optimizer_override_dict = {}
                layer_id = get_layer_id_for_vit(n, num_layers)

                if len(p.shape) == 1 or n.endswith(".bias"):
                    optimizer_override_dict["weight_decay_scale"] = 0

                if cfg.layer_decay > 0:
                    optimizer_override_dict["lr_scale"] = layer_scales[layer_id]
                p.optim_overrides = {"optimizer": optimizer_override_dict}

    @classmethod
    def build_model(cls, cfg: MaeImageClassificationConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def forward(
        self,
        imgs,
        labels=None,
    ):
        if self.training and self.mixup_fn is not None and labels is not None:
            imgs, labels = self.mixup_fn(imgs, labels)

        if self.linear_classifier:
            with torch.no_grad():
                x = self.model_forward(imgs)
        else:
            x = self.model_forward(imgs)

        if self.cfg.prediction_mode == PredictionMode.MEAN_POOLING:
            x = x.mean(dim=1)
        elif self.cfg.prediction_mode == PredictionMode.CLS_TOKEN:
            x = x[:, 0]
        elif self.cfg.prediction_mode == PredictionMode.LIN_SOFTMAX:
            dtype = x.dtype
            x = F.logsigmoid(x.float())
            x = torch.logsumexp(x + x, dim=1) - torch.logsumexp(x + 1e-6, dim=1)
            x = x.clamp(max=0)
            x = x - torch.log(-(torch.expm1(x)))
            x = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)
            x = x.to(dtype=dtype)
        else:
            raise Exception(f"unknown prediction mode {self.cfg.prediction_mode.name}")

        if self.fc_norm is not None:
            x = self.fc_norm(x)

        x = self.head(x)

        if labels is None:
            return x

        if self.training and self.mixup_fn is not None:
            loss = -labels * F.log_softmax(x.float(), dim=-1)
        else:
            loss = F.cross_entropy(
                x.float(),
                labels,
                label_smoothing=self.cfg.label_smoothing if self.training else 0,
                reduction="none",
            )

        result = {
            "losses": {"regression": loss},
            "sample_size": imgs.size(0),
        }

        if not self.training:
            with torch.no_grad():
                pred = x.argmax(-1)
                correct = (pred == labels).sum()
                result["correct"] = correct

        return result

    def model_forward(self, imgs):
        if self.d2v_multi:
            x = self.model.extract_features(
                imgs,
                mode="IMAGE",
                mask=False,
                remove_extra_tokens=(
                    self.cfg.prediction_mode != PredictionMode.CLS_TOKEN
                ),
            )["x"]
        else:
            x = self.model(imgs, predictions_only=True)
            if (
                "no_cls" not in self.model.cfg or not self.model.cfg.no_cls
            ) and not self.cfg.prediction_mode == PredictionMode.CLS_TOKEN:
                x = x[:, 1:]
        return x
