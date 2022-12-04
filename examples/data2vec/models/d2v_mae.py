# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from the BeiT implementation which can be found here:
# https://github.com/microsoft/unilm/tree/master/beit

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import math
import random
import torch.distributed as dist

from omegaconf import II

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.models import register_model
from .mae import (
    MaeModel,
    MaeConfig,
    RelativePositionBias,
    modify_relative_position_bias
)
from torch import nn
from fairseq.modules import SamePad2d, TransposeLast, LayerNorm
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class D2vMaeConfig(MaeConfig):
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )
    ema_anneal_end_step: int = II("optimization.max_update")

    instance_norm_target_layer: bool = False
    layer_norm_target_layer: bool = True
    layer_norm_targets: bool = True
    average_top_k_layers: int = 4

    dual_norm: bool = False

    decoder_type: str = "legacy"

    mae_conv_decoder_layers: int = 0
    mae_conv_decoder_no_act: bool = False
    mae_conv_decoder_kernel: int = 6
    mae_conv_decoder_groups: int = 1
    mae_conv_decoder_residual: bool = False
    mae_conv_decoder_no_ln: bool = True

    loss_scale: int = -1

    mask_type: str = "random"
    num_mask_patches: int = field(
        default=75,
        metadata={"help": "number of the visual tokens/patches need be masked"},
    )
    min_mask_patches_per_block: int = 16
    max_mask_patches_per_block: int = 196
    clone_batch: int = 1

    bert_init: bool = False

    shared_rel_pos_bias: bool = False

def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("d2v_mae", dataclass=D2vMaeConfig)
class D2vMaeModel(MaeModel):

    cfg: D2vMaeConfig

    def __init__(self, cfg: D2vMaeConfig, skip_ema=False):
        super().__init__(cfg)

        self.decoder_pred = torch.nn.Linear(
            cfg.decoder_embed_dim, cfg.embed_dim, bias=True
        )

        if cfg.decoder_type == "conv2d":
            ## we need to reconstruct the image; hence input and output dimensions needs to be the same.

            def make_block(in_dim):
                block = [
                    nn.Conv2d(
                        in_dim,
                        cfg.decoder_embed_dim,
                        kernel_size=cfg.mae_conv_decoder_kernel,
                        padding=cfg.mae_conv_decoder_kernel // 2,
                        groups=cfg.mae_conv_decoder_groups,
                    ),
                    SamePad2d(cfg.mae_conv_decoder_kernel),
                ]

                if not cfg.mae_conv_decoder_no_ln:
                    block.extend(
                        [
                            TransposeLast(tranpose_dim=-3),
                            LayerNorm(cfg.decoder_embed_dim, elementwise_affine=False),
                            TransposeLast(tranpose_dim=-3),
                        ]
                    )

                if not cfg.mae_conv_decoder_no_act:
                    block.append(nn.GELU())
                return nn.Sequential(*block)

            self.decoder_blocks = nn.Sequential(
                *[
                    make_block(
                        cfg.decoder_embed_dim
                        if not cfg.no_decoder_embed or i > 0
                        else cfg.embed_dim
                    )
                    for i in range(cfg.mae_conv_decoder_layers)
                ]
            )
        elif cfg.decoder_type == "conv1d":
            self.decoder_blocks = None
        else:
            if self.cfg.decoder_type != "legacy":
                raise NotImplementedError

        self.window_size = (
            cfg.input_size // self.patch_embed.patch_size[0],
            cfg.input_size // self.patch_embed.patch_size[1],
        )
        
        self.rel_pos_bias = None
        if self.cfg.shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.window_size, num_heads=cfg.num_heads
            )

        for pn, p in self.named_parameters():
            prefix = "decoder" if pn.startswith("decoder") else "encoder"
            if len(p.shape) == 1 or pn.endswith(".bias"):
                p.param_group = prefix + "_no_decay"
            else:
                p.param_group = prefix + "_with_decay"

        self.num_updates = 0

        if cfg.bert_init:
            from fairseq.modules.transformer_sentence_encoder import init_bert_params

            self.apply(init_bert_params)

        if not skip_ema:
            self.make_ema_teacher()

    @classmethod
    def build_model(cls, cfg: D2vMaeConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    @torch.no_grad()
    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
            add_missing_params=False,
            log_norms=True,
        )
        model_copy = D2vMaeModel(self.cfg, skip_ema=True)
        for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
            p_t.data.copy_(p_s.data)

        model_copy.remove_pretraining_modules()
        model_copy.requires_grad_(False)

        self.ema = EMAModule(
            model_copy,
            ema_config,
            copy_model=False,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates == num_updates
        ):
            p = next(self.parameters())
            device = p.device
            dtype = p.dtype

            p = next(self.ema.model.parameters())
            ema_device = p.device
            ema_dtype = p.dtype

            if ema_device != device or ema_dtype != dtype:
                logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
                self.ema.model = self.ema.model.to(dtype=dtype, device=device)
                for k, p in self.ema.fp32_params.items():
                    self.ema.fp32_params[k] = p.to(device=device)

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
                self.ema.step(self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if getattr(self, "ema", None) is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if getattr(self, "ema", None) is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def block_masking(self, x, num_masks, min_masks, max_masks):

        height, width = self.window_size
        N, L, D = x.shape  # batch, length, dim

        masks_bool_2d = np.zeros(shape=(N, height, width), dtype=np.bool_)

        for i in range(N):
            mask = masks_bool_2d[i]
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

        masks_1d = masks_bool_2d.reshape(N, -1)
        min_len = min([np.count_nonzero(m) for m in masks_1d])
        indices_1d = []

        for i, mask in enumerate(masks_1d):
            mask_idc = np.where(mask == 1)[0]
            if len(mask_idc) > min_len:
                mask_idc = np.random.choice(mask_idc, min_len, replace=False)
            masks_1d[i, :] = 0
            masks_1d[i, mask_idc] = 1
            unmasked_idc = np.where(masks_1d[i] == 0)[0]
            indices_1d.append(np.concatenate((unmasked_idc, mask_idc)))

        masks_1d = torch.from_numpy(masks_1d).to(x.device)
        new_m = masks_1d.unsqueeze(-1).repeat(1, 1, D)
        x_unmasked = x[~new_m].reshape(N, L - min_len, D)
        indices_1d = torch.from_numpy(np.array(indices_1d)).to(x.device)
        return x_unmasked, masks_1d, indices_1d

    def forward_encoder(self, x, mask_ratio, clone_batch=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        if self.cls_token is None:
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed[:, 1:, :]

        if self.cfg.clone_batch != 1 and clone_batch:
            x = x.repeat_interleave(self.cfg.clone_batch, 0)
        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            if self.cfg.mask_type == "random":
                x, mask, ids_restore = self.random_masking(x, mask_ratio)
            elif self.cfg.mask_type == "block":
                x, mask, ids_restore = self.block_masking(
                    x,
                    self.cfg.num_mask_patches,
                    self.cfg.min_mask_patches_per_block,
                    self.cfg.max_mask_patches_per_block,
                )
            else:
                raise NotImplementedError

        else:
            mask = ids_restore = None

        if self.cls_token is not None:
            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        layer_results = []

        if (self.cfg.dual_norm or not self.cfg.layer_norm_first) and self.norm is not None:
            x = self.norm(x)

        # apply Transformer blocks
        if self.cfg.w2v_block:
            x = x.transpose(0, 1)  # btc -> tbc

        for blk in self.blocks:
            if self.cfg.w2v_block:
                x, (z, lr) = blk(
                    x,
                    need_weights=False,
                    # alibi_bias=alibi_bias,
                )
                if self.cfg.end_of_block_targets:
                    layer_results.append(x.transpose(0, 1))
                else:
                    layer_results.append(lr.transpose(0, 1))
            elif self.cfg.alt_block:
                if self.rel_pos_bias is not None:
                    rel_pos_bias = self.rel_pos_bias()
                    rel_pos_bias = modify_relative_position_bias(rel_pos_bias, x.size(0), mask)
                else:
                    rel_pos_bias = None
                x, lr = blk(x, rel_pos_bias=rel_pos_bias, pos_mask=mask)
                layer_results.append(lr)
            elif self.cfg.alt_block2:
                x, lr = blk(x)
                layer_results.append(lr)
            else:
                x = blk(x)
                layer_results.append(x)

        if self.cfg.w2v_block:
            x = x.transpose(0, 1)  # btc -> tbc

        if (self.cfg.dual_norm or self.cfg.layer_norm_first) and self.norm is not None:
            x = self.norm(x)

        return x, mask, ids_restore, layer_results

    def forward_decoder(self, x, ids_restore):
        if self.cfg.decoder_type == "legacy":
            return self.forward_decoder_legacy(x, ids_restore)
        elif self.cfg.decoder_type == "conv2d":
            return self.forward_decoder_conv2d(x, ids_restore)
        else:
            raise NotImplementedError

    def forward_decoder_conv2d(self, x, ids_restore):
        if self.decoder_embed is not None:
            x = self.decoder_embed(
                x
            )  # this is just the mask tokens projected into new space
        # append mask tokens to sequence
        if self.mask_token is not None:
            mask_tokens = self.mask_token.repeat(
                x.shape[0], ids_restore.shape[1] + int(self.cls_token is not None) - x.shape[1], 1
            )
        else:
            mask_tokens = x.new_empty(
                x.size(0), ids_restore.shape[1] + int(self.cls_token is not None) - x.shape[1], x.size(-1)
            ).normal_(0, self.cfg.mask_noise_std)

        if self.cls_token is not None:
            x_ = torch.cat(
                [x[:, 1:, :], mask_tokens], dim=1
            )  # no cls token #i think you add back mask_tokens and then rearrange everything to the correct positions?
        else:
            x_ = torch.cat(
                [x, mask_tokens], dim=1
            )

        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = x_

        # add positional embedding
        x = x.reshape(
            x.size(0),
            int(np.sqrt(self.patch_embed.num_patches)),
            int(np.sqrt(self.patch_embed.num_patches)),
            -1,
        )

        if self.decoder_pos_embed is not None:
            reshaped_pos_embed = (
                self.decoder_pos_embed[:, 1:, :]
                .reshape(
                    int(np.sqrt(self.patch_embed.num_patches)),
                    int(np.sqrt(self.patch_embed.num_patches)),
                    -1,
                )
                .unsqueeze(0)
            )
            x = x + reshaped_pos_embed

        x = x.permute(0, 3, 1, 2)  # [N,H,W,D] => [N,D,H,W]

        for i, conv in enumerate(self.decoder_blocks):
            residual = x
            x = conv(x)
            if self.cfg.mae_conv_decoder_residual and (
                i > 0 or not self.cfg.no_decoder_embed
            ):
                x = x + residual

        x = x.permute(0, 2, 3, 1)  #  [N,D,H,W] => [N,H,W,D]
        x = x.reshape(
            x.size(0), self.patch_embed.num_patches, -1
        )  # [N, 14,14,d] => [N,14*14,d]
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]
        return x

    def forward_decoder_legacy(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, targets, pred):
        """
        targets: [N, L, D]
        pred: [N, L, D]
        """

        loss = F.mse_loss(pred.float(), targets.float(), reduction="none")

        if self.cfg.loss_scale < 0:
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        elif self.cfg.loss_scale == 0:
            loss = loss.sum(dim=-1) / math.sqrt(pred.size(-1))
        else:
            loss = loss.sum(dim=-1) * self.cfg.loss_scale

        return loss

    def forward(self, imgs, predictions_only=False):
        latent, mask, ids_restore, _ = self.forward_encoder(
            imgs,
            self.mask_ratio if not predictions_only else 0,
            clone_batch=(self.cfg.clone_batch != 1) and not predictions_only,
        )

        if predictions_only:
            return latent

        with torch.no_grad():
            _, _, _, targets = self.ema.model.forward_encoder(
                imgs, mask_ratio=0, clone_batch=False
            )
            mask = mask.bool()

            targets = targets[-self.cfg.average_top_k_layers :]

            if self.cls_token is not None:
                targets = [t[:, 1:] for t in targets]

            if self.cfg.instance_norm_target_layer:
                targets = [
                    F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2)
                    for tl in targets
                ]

            targets = [
                target.repeat_interleave(self.cfg.clone_batch, 0) for target in targets
            ]

            targets = [t[mask] for t in targets]

            if self.cfg.layer_norm_target_layer:
                targets = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in targets]

            targets = sum(targets) / len(targets)

            if self.cfg.layer_norm_targets:
                targets = F.layer_norm(targets, targets.shape[-1:])

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, D]
        pred = pred[mask]

        loss = self.forward_loss(targets, pred)
        sample_size = loss.numel()

        result = {
            "losses": {"regression": loss},
            "sample_size": sample_size,
            "logs": {},
        }
        if mask is not None:
            result["logs"]["masked_pct"] = mask.sum() / (mask.size(0) * mask.size(1))

        with torch.no_grad():
            result["pred_var"] = self.compute_var(pred.float())
            result["target_var"] = self.compute_var(targets.float())
            if self.ema is not None:
                result["ema_decay"] = self.ema.get_decay() * 1000
                for k, v in self.ema.logs.items():
                    result[k] = v

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

    def remove_pretraining_modules(self):
        super().remove_pretraining_modules()
