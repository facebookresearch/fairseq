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
from fairseq.models.roberta.model import RobertaClassificationHead

from examples.data2vec.data.modality import Modality


logger = logging.getLogger(__name__)


@dataclass
class Data2VecTextClassificationConfig(FairseqDataclass):
    pooler_dropout: float = 0.0
    pooler_activation_fn: str = "tanh"
    quant_noise_pq: int = 0
    quant_noise_pq_block_size: int = 8
    spectral_norm_classification_head: bool = False

    model_path: str = MISSING
    no_pretrained_weights: bool = False

    pretrained_model_args: Any = None


@register_model(
    "data2vec_text_classification", dataclass=Data2VecTextClassificationConfig
)
class Data2VecTextClassificationModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecTextClassificationConfig):
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

        task = tasks.setup_task(pretrained_args.task)
        model = task.build_model(pretrained_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        self.model = model

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        self.classification_heads = nn.ModuleDict()


    def load_model_weights(self, state, model, cfg):
        for k in list(state["model"].keys()):
            if (
                k.startswith("shared_decoder") or
                k.startswith("_ema") or
                "decoder" in k
            ):
                logger.info(f"Deleting {k} from checkpoint")
                del state["model"][k]
        model.load_state_dict(state["model"], strict=True)

    @classmethod
    def build_model(cls, cfg: Data2VecTextClassificationConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        embed_dim = self.cfg.pretrained_model_args.model.embed_dim
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=embed_dim,
            inner_dim=inner_dim or embed_dim,
            num_classes=num_classes,
            activation_fn=self.cfg.pooler_activation_fn,
            pooler_dropout=self.cfg.pooler_dropout,
            q_noise=self.cfg.quant_noise_pq,
            qn_block_size=self.cfg.quant_noise_pq_block_size,
            do_spectral_norm=self.cfg.spectral_norm_classification_head,
        )

    def forward(
        self,
        source,
        id,
        padding_mask,
        features_only=True,
        remove_extra_tokens=True,
        classification_head_name=None,
    ):
        encoder_out = self.model(
            source,
            id=id,
            mode=Modality.TEXT,
            padding_mask=padding_mask,
            mask=False,
            features_only=features_only,
            remove_extra_tokens=remove_extra_tokens
        )
        logits = self.classification_heads[classification_head_name](encoder_out["x"])
        return logits, encoder_out
