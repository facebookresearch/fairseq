# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from the BeiT implementation which can be found here:
# https://github.com/microsoft/unilm/tree/master/beit

import logging
import math
import numpy as np
import random

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model


logger = logging.getLogger(__name__)


@dataclass
class Data2VecVisionConfig(FairseqDataclass):
    layer_scale_init_value: float = field(
        default=1e-4, metadata={"help": "rescale layer outputs, 0 to disable"}
    )
    num_mask_patches: int = field(
        default=75,
        metadata={"help": "number of the visual tokens/patches need be masked"},
    )
    min_mask_patches_per_block: int = 16
    max_mask_patches_per_block: int = 196
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    shared_rel_pos_bias: bool = True

    drop_path: float = 0.1
    attention_dropout: float = 0.0

    depth: int = 12
    embed_dim: int = 768
    num_heads: int = 12
    mlp_ratio: int = 4

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = True
    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("data2vec_vision", dataclass=Data2VecVisionConfig)
class Data2VecVisionModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecVisionConfig):
        super().__init__()
        self.cfg = cfg

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = (
            cfg.loss_scale
            if cfg.loss_scale is not None
            else 1 / math.sqrt(cfg.embed_dim)
        )

        self.patch_embed = PatchEmbed(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )

        patch_size = self.patch_embed.patch_size
        self.window_size = (
            cfg.image_size // patch_size[0],
            cfg.image_size // patch_size[1],
        )

        self.cls_emb = nn.Parameter(torch.FloatTensor(1, 1, cfg.embed_dim))
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, cfg.embed_dim))

        nn.init.trunc_normal_(self.cls_emb, 0.02)
        nn.init.trunc_normal_(self.mask_emb, 0.02)

        self.encoder = TransformerEncoder(cfg, self.patch_embed.patch_shape)

        self.final_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.num_updates = 0

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_proj is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecVisionConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def make_mask(self, bsz, num_masks, min_masks, max_masks):
        height, width = self.window_size

        masks = np.zeros(shape=(bsz, height, width), dtype=np.int)

        for i in range(bsz):
            mask = masks[i]
            mask_count = 0

            min_aspect = 0.3
            max_aspect = 1 / min_aspect
            log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

            def _mask(mask, max_mask_patches):
                delta = 0
                for attempt in range(10):
                    target_area = random.uniform(min_masks, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < width and h < height:
                        top = random.randint(0, height - h)
                        left = random.randint(0, width - w)

                        num_masked = mask[top : top + h, left : left + w].sum()
                        # Overlap
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1

                        if delta > 0:
                            break
                return delta

            while mask_count < num_masks:
                max_mask_patches = min(num_masks - mask_count, max_masks)

                delta = _mask(mask, max_mask_patches)
                if delta == 0:
                    break
                else:
                    mask_count += delta

        return torch.from_numpy(masks)

    def forward(
        self,
        img,
        mask: bool = True,
        layer_results: bool = False,
    ):
        x = self.patch_embed(img)
        batch_size, seq_len, _ = x.size()

        if mask:
            mask_indices = self.make_mask(
                img.size(0),
                self.cfg.num_mask_patches,
                self.cfg.min_mask_patches_per_block,
                self.cfg.max_mask_patches_per_block,
            )
            bool_mask = mask_indices.view(mask_indices.size(0), -1).bool()
        else:
            mask_indices = bool_mask = None

        cls_tokens = self.cls_emb.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.ema is not None:
            with torch.no_grad():
                self.ema.model.eval()

                if self.cfg.ema_transformer_only:
                    y = self.ema.model(
                        x,
                        layer_results="end" if self.cfg.end_of_block_targets else "fc",
                    )
                else:
                    y = self.ema.model(
                        img,
                        mask=False,
                        layer_results=True,
                    )

            y = y[-self.cfg.average_top_k_layers :]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                y = [tl.transpose(1, 2) for tl in y]  # BTC -> BCT
                permuted = True

            if self.cfg.batch_norm_target_layer:
                y = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in y
                ]

            if self.cfg.instance_norm_target_layer:
                y = [F.instance_norm(tl.float()) for tl in y]

            if permuted:
                y = [tl.transpose(1, 2) for tl in y]  # BCT -> BTC

            if self.cfg.layer_norm_target_layer:
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

            y = sum(y) / len(y)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            y = y[bool_mask].float()

        if mask_indices is not None:
            mask_token = self.mask_emb.expand(batch_size, seq_len, -1)
            w = mask_indices.view(mask_indices.size(0), -1, 1).type_as(mask_token)
            x[:, 1:] = x[:, 1:] * (1 - w) + mask_token * w

        if layer_results:
            enc_layer_results = "end" if self.cfg.end_of_block_targets else "fc"
        else:
            enc_layer_results = None

        x = self.encoder(x, layer_results=enc_layer_results)
        if layer_results or mask_indices is None:
            return x

        x = x[bool_mask].float()

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta).sum(
                dim=-1
            )

        if self.loss_scale > 0:
            loss = loss * self.loss_scale

        result = {
            "losses": {"regression": loss.sum()},
            "sample_size": loss.numel(),
            "target_var": self.compute_var(y),
            "pred_var": self.compute_var(x),
            "ema_decay": self.ema.get_decay() * 1000,
        }
        return result

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        self.encoder.norm = nn.Identity()
        self.mask_emb = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # BCHW -> BTC
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            assert 1==2
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        print("attn.size() :", attn.size())
        print("rel_pos_bias.size() :", rel_pos_bias.size())
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        print("self.window_size :", self.window_size)
        print("self.num_relative_distance :", self.num_relative_distance)
        print("self.relative_position_index :", self.relative_position_index.size(), self.relative_position_index)
        print("relative_position_bias.size(), relative_position_bias :",relative_position_bias.size(), relative_position_bias)
        print("self.relative_position_bias_table.size(), self.relative_position_bias_table :",self.relative_position_bias_table.size(), self.relative_position_bias_table)
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        window_size=None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        print("inside block :", x.size())
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            fc_feature = self.drop_path(self.mlp(self.norm2(x)))
            x = x + fc_feature
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            )
            fc_feature = self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            x = x + fc_feature
        return x, fc_feature


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: Data2VecVisionConfig, patch_shape):
        super().__init__()

        self.rel_pos_bias = None
        if cfg.shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=patch_shape, num_heads=cfg.num_heads
            )

        dpr = [
            x.item() for x in torch.linspace(0, cfg.drop_path, cfg.depth)
        ]  # stochastic depth decay rule

        print("TransformerEncoder > patch_shape :", patch_shape)
        self.blocks = nn.ModuleList(
            Block(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                attn_drop=cfg.attention_dropout,
                drop_path=dpr[i],
                init_values=cfg.layer_scale_init_value,
                window_size=patch_shape if not cfg.shared_rel_pos_bias else None,
            )
            for i in range(cfg.depth)
        )

        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.apply(self.init_weights)
        self.fix_init_weight()

    def init_weights(self, m):
        std = 0.02
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp[2].weight.data, layer_id + 1)

    def extract_features(self, x, layer_results):

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        z = []
        for i, blk in enumerate(self.blocks):
            x, fc_feature = blk(x, rel_pos_bias=rel_pos_bias)
            if layer_results == "end":
                z.append(x)
            elif layer_results == "fc":
                z.append(fc_feature)

        return z if layer_results else self.norm(x)

    def forward(self, x, layer_results=None):
        x = self.extract_features(x, layer_results=layer_results)
        if layer_results:
            return [z[:, 1:] for z in x]

        x = x[:, 1:]
        return x
