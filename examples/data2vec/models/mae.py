# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from the BeiT implementation which can be found here:
# https://github.com/microsoft/unilm/tree/master/beit

import logging
from dataclasses import dataclass
from functools import partial

from timm.models.vision_transformer import PatchEmbed, Block

import torch
import torch.nn as nn

import numpy as np

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer

from apex.normalization import FusedLayerNorm
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class MaeConfig(FairseqDataclass):
    input_size: int = 224
    in_chans: int = 3
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: int = 4
    norm_eps: float = 1e-6

    drop_path_rate: float = 0.0

    mask_ratio: float = 0.75
    norm_pix_loss: bool = True

    w2v_block: bool = False
    alt_block: bool = False
    alt_block2: bool = False
    alt_attention: bool = False
    block_dropout: float = 0
    attention_dropout: float = 0
    activation_dropout: float = 0
    layer_norm_first: bool = False

    fused_ln: bool = True
    end_of_block_targets: bool = True

    no_decoder_embed: bool = False
    no_decoder_pos_embed: bool = False
    mask_noise_std: float = 0

    single_qkv: bool = False
    use_rel_pos_bias: bool = False
    no_cls: bool = False


def modify_relative_position_bias(orig_bias, bsz, mask):
    if mask is None:
        return orig_bias.unsqueeze(0).repeat(
            bsz, 1, 1, 1
        )  # heads x seq_len x seq_len => bsz x heads x seq_len x seq_len
    heads, max_seq_len, max_seq_len = orig_bias.shape  # includes CLS token
    mask_for_rel_pos_bias = torch.cat(
        (torch.zeros(bsz, 1, dtype=mask.dtype, device=mask.device), mask), dim=1
    ).bool()  # bsz x seqlen (add CLS token)
    unmasked_for_rel_pos_bias = ~mask_for_rel_pos_bias
    unmasked_for_rel_pos_bias = unmasked_for_rel_pos_bias.unsqueeze(1).repeat(
        1, heads, 1
    )  # bsz x seq_len => bsz x heads x seq_len
    b_t_t_rel_pos_bias = orig_bias.unsqueeze(0).repeat(
        bsz, 1, 1, 1
    )  # heads x seq_len x seq_len => bsz x heads x seq_len x seq_len
    b_t_t_rel_pos_bias = b_t_t_rel_pos_bias.masked_select(
        unmasked_for_rel_pos_bias.unsqueeze(-1)
    )
    b_t_t_rel_pos_bias = b_t_t_rel_pos_bias.view(bsz, heads, -1, max_seq_len)
    new_len = b_t_t_rel_pos_bias.size(-2)
    b_t_t_rel_pos_bias = b_t_t_rel_pos_bias.masked_select(
        unmasked_for_rel_pos_bias.unsqueeze(-2)
    )
    b_t_t_rel_pos_bias = b_t_t_rel_pos_bias.view(bsz, heads, new_len, new_len)
    return b_t_t_rel_pos_bias


class AltBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        ffn_targets=False,
        use_rel_pos_bias=False,
        window_size=None,
        alt_attention=False,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets

        from timm.models.vision_transformer import Attention, DropPath, Mlp

        self.norm1 = norm_layer(dim)
        self.use_rel_pos_bias = use_rel_pos_bias
        if use_rel_pos_bias:
            self.attn = AltAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                window_size=window_size,
            )
        else:
            if alt_attention:
                from .multi.modules import AltAttention as AltAttention2
                self.attn = AltAttention2(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
            else:
                self.attn = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, rel_pos_bias=None, pos_mask=None):
        if self.layer_norm_first:
            if self.use_rel_pos_bias:
                x = x + self.drop_path(
                    self.attn(
                        self.norm1(x), rel_pos_bias=rel_pos_bias, pos_mask=pos_mask
                    )
                )
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
            t = self.mlp(self.norm2(x))
            x = x + self.drop_path(t)
            if not self.ffn_targets:
                t = x
            return x, t
        else:
            if self.use_rel_pos_bias:
                x = x + self.drop_path(
                    self.attn(x, rel_pos_bias=rel_pos_bias, pos_mask=pos_mask)
                )
            else:
                x = x + self.drop_path(self.attn(x))
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(x))
            if not self.ffn_targets:
                t = x
            return x, t


class AltAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
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
        self.scale = qk_scale or head_dim ** -0.5

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

    def forward(self, x, rel_pos_bias=None, pos_mask=None):
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
            attn = attn + modify_relative_position_bias(
                relative_position_bias, x.size(0), pos_mask
            )

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
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


@register_model("mae", dataclass=MaeConfig)
class MaeModel(BaseFairseqModel):
    def __init__(self, cfg: MaeConfig):
        super().__init__()
        self.cfg = cfg

        self.mask_ratio = cfg.mask_ratio

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            cfg.input_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) if not cfg.no_cls else None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + int(not cfg.no_cls), cfg.embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        norm_layer = partial(nn.LayerNorm, eps=cfg.norm_eps)

        dpr = [
            x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.depth)
        ]  # stochastic depth decay rule

        def make_block(drop_path):
            if cfg.w2v_block:
                return TransformerSentenceEncoderLayer(
                    embedding_dim=cfg.embed_dim,
                    ffn_embedding_dim=cfg.embed_dim * cfg.mlp_ratio,
                    num_attention_heads=cfg.num_heads,
                    dropout=cfg.block_dropout,
                    attention_dropout=cfg.attention_dropout,
                    activation_dropout=cfg.activation_dropout,
                    activation_fn="gelu",
                    layer_norm_first=cfg.layer_norm_first,
                    drop_path=drop_path,
                    norm_eps=1e-6,
                    single_qkv=cfg.single_qkv,
                    fused_ln=cfg.fused_ln,
                )
            elif cfg.alt_block:
                window_size = (
                    cfg.input_size // self.patch_embed.patch_size[0],
                    cfg.input_size // self.patch_embed.patch_size[1],
                )
                return AltBlock(
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    layer_norm_first=cfg.layer_norm_first,
                    ffn_targets=not cfg.end_of_block_targets,
                    use_rel_pos_bias=cfg.use_rel_pos_bias,
                    window_size=window_size
                    if (self.cfg.use_rel_pos_bias and not self.cfg.shared_rel_pos_bias)
                    else None,
                    alt_attention=cfg.alt_attention,
                )
            elif cfg.alt_block2:
                from .multi.modules import AltBlock as AltBlock2
                return AltBlock2(
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    layer_norm_first=cfg.layer_norm_first,
                    ffn_targets=not cfg.end_of_block_targets,
                )
            else:
                return Block(
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                )

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])
        self.norm = norm_layer(cfg.embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = (
            nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim, bias=True)
            if not cfg.no_decoder_embed
            else None
        )

        self.mask_token = (
            nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    cfg.decoder_embed_dim
                    if not cfg.no_decoder_embed
                    else cfg.embed_dim,
                )
            )
            if cfg.mask_noise_std <= 0
            else None
        )

        self.decoder_pos_embed = (
            nn.Parameter(
                torch.zeros(
                    1,
                    num_patches + 1,
                    cfg.decoder_embed_dim
                    if not cfg.no_decoder_embed
                    else cfg.embed_dim,
                ),
                requires_grad=False,
            )
            if not cfg.no_decoder_pos_embed
            else None
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    cfg.decoder_embed_dim,
                    cfg.decoder_num_heads,
                    cfg.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(cfg.decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(cfg.decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            cfg.decoder_embed_dim, cfg.patch_size ** 2 * cfg.in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = cfg.norm_pix_loss

        self.initialize_weights()

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias"):
                p.param_group = "no_decay"
            else:
                p.param_group = "with_decay"

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=not self.cfg.no_cls,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.decoder_pos_embed is not None:
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                int(self.patch_embed.num_patches ** 0.5),
                cls_token=not self.cfg.no_cls,
            )
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
            )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=0.02)

        if self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, FusedLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore  # x_masked is actually unmasked x

    @classmethod
    def build_model(cls, cfg: MaeConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # if self.cls_token is not None:
        #     x = x + self.pos_embed
        # else:
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = ids_restore = None

        # append cls token
        if self.cls_token is not None:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        if self.cls_token is not None:
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        else:
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token

        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        if self.cls_token is not None:
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.cls_token is not None:
            # remove cls token
            x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum()
        return loss, mask.sum()

    def forward(self, imgs, predictions_only=False):
        latent, mask, ids_restore = self.forward_encoder(
            imgs, self.mask_ratio if not predictions_only else 0
        )

        if predictions_only:
            return latent

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, sample_size = self.forward_loss(imgs, pred, mask)

        result = {
            "losses": {"regression": loss},
            "sample_size": sample_size,
        }
        return result

    def remove_pretraining_modules(self):
        self.decoder_embed = None
        self.decoder_blocks = None
        self.decoder_norm = None
        self.decoder_pos_embed = None
        self.decoder_pred = None
        self.mask_token = None
        if self.cfg.layer_norm_first:
            self.norm = None
