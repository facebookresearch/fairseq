# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Any
from functools import partial
import numpy as np

from omegaconf import MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .multi.modules import AltBlock

logger = logging.getLogger(__name__)


@dataclass
class Data2VecVisualizerConfig(FairseqDataclass):

    depth: int = 8
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 8
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.0
    post_mlp_drop: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 512
    mlp_ratio: float = 4
    layer_norm_first: bool = False

    normalize_pixels: bool = True
    from_decoder: bool = False
    average_targets: bool = False
    target_layer: int = -1
    end_of_block: bool = True

    linear_blocks: bool = False

    model_path: str = MISSING
    model_args: Any = None


@register_model("data2vec_visualizer", dataclass=Data2VecVisualizerConfig)
class Data2VecVisualizerModel(BaseFairseqModel):
    def _init_weights(self, m):

        try:
            from apex.normalization import FusedLayerNorm

            fn = FusedLayerNorm
        except:
            fn = nn.LayerNorm

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def __init__(self, cfg: Data2VecVisualizerConfig):
        super().__init__()
        self.cfg = cfg

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            if cfg.linear_blocks:
                d = cfg.embed_dim if dim is None else dim
                return nn.Sequential(nn.Linear(d, d), make_layer_norm(d), nn.GELU())

            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
            )

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)
        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first and len(self.blocks) > 0:
            self.norm = make_layer_norm(cfg.embed_dim)

        self.apply(init_bert_params)

        self.target_model, self.target_model_cfg = self.make_target_model(cfg)
        self.target_model.requires_grad_(False)

        self.in_proj = (
            nn.Linear(self.target_model_cfg.embed_dim, cfg.embed_dim)
            if self.target_model_cfg.embed_dim != cfg.embed_dim
            else None
        )
        self.recon_proj = nn.Linear(cfg.embed_dim, self.target_model_cfg.embed_dim)

        self.top_k = self.target_model_cfg.average_top_k_layers
        self.num_extra = self.target_model_cfg.modalities.image.num_extra_tokens

    def make_target_model(self, cfg):
        state = None
        if cfg.model_args is None:
            arg_overrides = None
            if not self.cfg.average_targets:
                arg_overrides = {
                    "end_of_block_targets": self.cfg.end_of_block,
                }
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.model_path, arg_overrides
            )
            model_args = state["cfg"]
            model_args.criterion = None
            model_args.lr_scheduler = None

            cfg.model_args = model_args

            logger.info(model_args)
        task = tasks.setup_task(model_args.task)
        model = task.build_model(model_args.model, from_checkpoint=True)

        if state is not None:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules(keep_decoder=cfg.from_decoder)

        return model, model_args.model

    @classmethod
    def build_model(cls, cfg: Data2VecVisualizerConfig, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def unpatchify(self, x):
        mod_enc = self.target_model.modality_encoders["IMAGE"]
        return mod_enc.unpatchify(x)

    def forward(
        self,
        source,
        padding_mask=None,
        precomputed_mask=None,
    ):

        model_args = {
            "source": source,
            "padding_mask": padding_mask,
            "precomputed_mask": precomputed_mask,
            "features_only": True,
            "mask": precomputed_mask is not None,
            "remove_extra_tokens": False,
            "force_remove_masked": True,
        }

        with torch.no_grad():
            self.target_model.eval()
            mod_enc = self.target_model.modality_encoders["IMAGE"]

            target = mod_enc.patchify(source)

            if self.cfg.normalize_pixels:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            tm_res = self.target_model(**model_args)

            if self.cfg.from_decoder:
                x = self.target_model.forward_decoder(
                    tm_res["x"],
                    mod_enc,
                    mod_enc.decoder,
                    tm_res["mask"],
                )
            elif self.cfg.average_targets:
                lrs = tm_res["layer_results"]
                if self.num_extra > 0:
                    lrs = [lr[:, self.num_extra :] for lr in lrs]
                x = self.target_model.make_targets(lrs, self.top_k).type_as(source)
            elif self.cfg.target_layer >= 0:
                assert precomputed_mask is None
                x = tm_res["layer_results"][self.cfg.target_layer][:, self.num_extra :]
            else:
                assert precomputed_mask is None
                x = tm_res["x"][:, self.num_extra :]

        x = self.dropout_input(x)

        if self.in_proj is not None:
            x = self.in_proj(x)

        for blk in self.blocks:
            if self.cfg.linear_blocks:
                x = blk(x)
            else:
                x, _ = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        x = self.recon_proj(x)
        loss = F.mse_loss(x.float(), target.float(), reduction="none")

        result = {
            "x": x,
            "losses": {"recon": loss},
            "sample_size": loss.numel() // x.size(-1),
        }

        return result
