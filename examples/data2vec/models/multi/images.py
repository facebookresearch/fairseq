# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from timm.models.layers import to_2tuple
from fairseq.tasks import FairseqTask
from ..mae import get_2d_sincos_pos_embed, PatchEmbed
from .base import (
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_alibi_bias,
    MaskSeed,
    MaskInfo,
)
from .modules import (
    BlockEncoder,
    Decoder2d,
    FixedPositionalEncoder,
    D2vDecoderConfig,
    TransformerDecoder,
    EncDecTransformerDecoder,
    CBlock,
    ConvMAEPatchEmbed,
)
from ...tasks.multimodal import Modality


@dataclass
class D2vImageConfig(D2vModalityConfig):
    type: Modality = Modality.IMAGE

    input_size: int = 224
    in_chans: int = 3
    patch_size: int = 16
    embed_dim: int = 768

    fix_masks: bool = False
    exact_mask_pct: bool = False

    unmask_focal: bool = False
    focal_length: int = 1

    alibi_dims: int = 2
    alibi_distance: str = "manhattan"

    fixed_positions: bool = True
    conv_pos_cfg: Optional[D2vDecoderConfig] = None

    transformer_decoder: bool = False
    enc_dec_transformer: bool = False

    conv_mae: bool = False
    conv_mae_multiscale: bool = True
    conv_mae_masking: bool = True


class ImageEncoder(ModalitySpecificEncoder):

    modality_cfg: D2vImageConfig

    def __init__(
        self,
        modality_cfg: D2vImageConfig,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
    ):

        img_size = to_2tuple(modality_cfg.input_size)
        patch_size = to_2tuple(modality_cfg.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        if modality_cfg.conv_mae:
            local_encoder = ConvMAEPatchEmbed(
                modality_cfg.input_size,
                modality_cfg.patch_size,
                modality_cfg.in_chans,
                modality_cfg.embed_dim,
            )
        else:
            local_encoder = PatchEmbed(
                modality_cfg.input_size,
                modality_cfg.patch_size,
                modality_cfg.in_chans,
                modality_cfg.embed_dim,
            )

        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if modality_cfg.embed_dim != embed_dim and not modality_cfg.conv_mae:
            local_encoder = nn.Sequential(
                local_encoder,
                nn.Linear(modality_cfg.embed_dim, embed_dim),
            )

        project_features = nn.Identity()

        if modality_cfg.conv_mae:
            pos_embed = nn.Parameter(torch.zeros(1, 14 * 14, 768), requires_grad=False)
            side_n = 14
        else:
            pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim), requires_grad=False
            )

            side_n = int(num_patches ** 0.5)

        emb = get_2d_sincos_pos_embed(
            pos_embed.shape[-1],
            side_n,
            cls_token=False,
        )
        pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))
        fixed_positional_encoder = (
            FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None
        )
        relative_positional_encoder = (
            Decoder2d(modality_cfg.conv_pos_cfg, embed_dim, side_n, side_n)
            if modality_cfg.conv_pos_cfg is not None
            else None
        )

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )

        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        if modality_cfg.transformer_decoder:
            if modality_cfg.enc_dec_transformer:
                decoder = EncDecTransformerDecoder(modality_cfg.decoder, embed_dim)
            else:
                dec_enc = BlockEncoder(
                    nn.ModuleList(
                        make_block(0, modality_cfg.decoder.decoder_dim, 8)
                        for _ in range(modality_cfg.decoder.decoder_layers)
                    ),
                    None,
                    layer_norm_first,
                    0,
                    0,
                )
                decoder = TransformerDecoder(modality_cfg.decoder, embed_dim, dec_enc)
        else:
            decoder = (
                Decoder2d(modality_cfg.decoder, embed_dim, side_n, side_n)
                if modality_cfg.decoder is not None
                else None
            )

        alibi_bias_fn = partial(
            get_alibi_bias,
            alibi_biases=alibi_biases,
            heads=modality_cfg.num_alibi_heads,
            dims=modality_cfg.alibi_dims,
            distance=modality_cfg.alibi_distance,
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=relative_positional_encoder,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

        if modality_cfg.conv_mae:
            assert modality_cfg.patch_size == 4
            assert modality_cfg.embed_dim == 256
            assert modality_cfg.num_extra_tokens == 0

            self.patch_embed2 = ConvMAEPatchEmbed(
                img_size=56, patch_size=2, in_chans=256, embed_dim=384
            )
            w = self.patch_embed2.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            self.patch_embed3 = ConvMAEPatchEmbed(
                img_size=28, patch_size=2, in_chans=384, embed_dim=768
            )
            w = self.patch_embed3.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            self.blocks1 = nn.ModuleList(
                [
                    CBlock(
                        dim=modality_cfg.embed_dim,
                        num_heads=12,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for _ in range(2)
                ]
            )
            self.blocks2 = nn.ModuleList(
                [
                    CBlock(
                        dim=384,
                        num_heads=12,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for _ in range(2)
                ]
            )
            self.norm = norm_layer(768)

            self.patch_embed4 = nn.Linear(768, 768)

            if modality_cfg.conv_mae_multiscale:
                self.stage1_output_decode = nn.Conv2d(
                    modality_cfg.embed_dim, 768, 4, stride=4
                )
                self.stage2_output_decode = nn.Conv2d(384, 768, 2, stride=2)
            else:
                self.stage1_output_decode = self.stage2_output_decode = None

    def reset_parameters(self):
        super().reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    @torch.no_grad()
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.modality_cfg.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    @torch.no_grad()
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.modality_cfg.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        shape=None,
        precomputed_mask=None,
    ):
        mlen = self.modality_cfg.mask_length
        if mlen <= 1:
            return super().compute_mask(
                x, padding_mask, mask_seed, apply, precomputed_mask
            )

        if precomputed_mask is not None:
            mask = precomputed_mask
        else:
            if shape is not None:
                B, L, D = shape
            else:
                B, L, D = x.shape

            bsz = B
            d = int(L ** 0.5)
            mask_prob = self.modality_cfg.mask_prob

            if (
                self.modality_cfg.mask_prob_min is not None
                and self.modality_cfg.mask_prob_min >= 0
                and self.modality_cfg.mask_prob_min < mask_prob
            ):
                mask_prob = np.random.uniform(
                    self.modality_cfg.mask_prob_min, mask_prob
                )

            if self.modality_cfg.inverse_mask:
                mask_prob = 1 - mask_prob

            mask = torch.zeros((bsz, d, d))
            mask_inds = torch.randint(
                0,
                d * d,
                size=(
                    bsz,
                    int(d ** 2 * (mask_prob / mlen ** 2)),
                ),
            )
            mask.view(bsz, -1).scatter_(1, mask_inds, 1)
            centers = mask.nonzero(as_tuple=True)

            inds = ([], [], [])

            offset = mlen // 2
            for i in range(mlen):
                for j in range(mlen):
                    k1 = i - offset
                    k2 = j - offset
                    inds[0].append(centers[0])
                    inds[1].append(centers[1] + k1)
                    inds[2].append(centers[2] + k2)

            i0 = torch.cat(inds[0])
            i1 = torch.cat(inds[1]).clamp_(min=0, max=d - 1)
            i2 = torch.cat(inds[2]).clamp_(min=0, max=d - 1)

            mask[(i0, i1, i2)] = 1

            mask = mask.view(bsz, -1)

            if self.modality_cfg.fix_masks:
                n_masks = mask.sum(dim=-1)

                if self.modality_cfg.exact_mask_pct:
                    target_len = int(L * mask_prob)
                else:
                    add_mask = self.modality_cfg.add_masks
                    if add_mask and self.modality_cfg.remove_masks:
                        target_len = (n_masks.max() + n_masks.min()) / 2
                    else:
                        if self.modality_cfg.inverse_mask:
                            add_mask = not add_mask

                        if add_mask:
                            target_len = n_masks.max()
                        else:
                            target_len = n_masks.min()

                for n, m in zip(n_masks, mask):
                    if n > target_len:
                        to_unmask = torch.multinomial(
                            m, int(n - target_len), replacement=False
                        )
                        m[to_unmask] = 0
                    elif n < target_len:
                        to_mask = torch.multinomial(
                            (1 - m), int(target_len - n), replacement=False
                        )
                        m[to_mask] = 1

            mask = mask.to(x.device)
            if self.modality_cfg.inverse_mask:
                mask = 1 - mask

        mask_info = self.make_maskinfo(x, mask, shape)
        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def decoder_input(self, x, mask_info):
        if (
            not self.modality_cfg.transformer_decoder
            or not self.modality_cfg.enc_dec_transformer
        ):
            return super().decoder_input(x, mask_info)

        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        kv = x[:, self.modality_cfg.num_extra_tokens :]

        assert self.fixed_positional_encoder is not None
        pos = self.fixed_positional_encoder(x, None).expand(x.size(0), -1, -1)

        mask = mask_info.mask.bool()
        if self.modality_cfg.decoder.add_positions_all:
            kv = kv + pos[~mask].view(kv.shape)

        q = pos[mask].view(x.size(0), -1, x.size(-1))

        return q, kv

    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        if not self.modality_cfg.conv_mae:
            return super(ImageEncoder, self).contextualized_features(
                x,
                padding_mask,
                mask,
                remove_masked,
                clone_batch,
                mask_seeds,
                precomputed_mask,
            )

        assert padding_mask is None

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        orig_B, orig_T, _, _ = x.shape

        bin_mask1 = bin_mask2 = None

        mask_info = None
        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)

            if self.modality_cfg.conv_mae_masking or self.modality_cfg.mask_length == 1:
                mask_info = self.convmae_random_masking(x, self.modality_cfg.mask_prob)
            else:
                _, mask_info = self.compute_mask(
                    x,
                    padding_mask,
                    mask_seed=mask_seeds,
                    apply=False,
                    shape=(x.shape[0], self.patch_embed3.num_patches, 768),
                )

            bin_mask1 = 1 - mask_info.mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(
                1, 1, 1, 16
            ).reshape(-1, 14, 14, 4, 4).permute(0, 1, 3, 2, 4).reshape(
                x.shape[0], 56, 56
            ).unsqueeze(
                1
            )
            bin_mask2 = 1 - mask_info.mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(
                1, 1, 1, 4
            ).reshape(-1, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(
                x.shape[0], 28, 28
            ).unsqueeze(
                1
            )

        for blk in self.blocks1:
            x = blk(x, bin_mask1)

        stage1_embed = None
        if self.stage1_output_decode is not None:
            stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, bin_mask2)

        stage2_embed = None
        if self.stage2_output_decode is not None:
            stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed3(x)

        x = x.flatten(2).permute(0, 2, 1)

        x = self.patch_embed4(x)

        x = x + self.fixed_positional_encoder(x, None)

        if mask:
            x = torch.gather(
                x,
                dim=1,
                index=mask_info.ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]),
            )
            if stage1_embed is None:
                stage1_embed = 0
            else:
                stage1_embed = torch.gather(
                    stage1_embed,
                    dim=1,
                    index=mask_info.ids_keep.unsqueeze(-1).repeat(
                        1, 1, stage1_embed.shape[-1]
                    ),
                )
            if stage2_embed is None:
                stage2_embed = 0
            else:
                stage2_embed = torch.gather(
                    stage2_embed,
                    dim=1,
                    index=mask_info.ids_keep.unsqueeze(-1).repeat(
                        1, 1, stage2_embed.shape[-1]
                    ),
                )

        x = self.context_encoder(
            x,
            padding_mask,
            None,
            None,
        )

        if mask:
            x = x + stage1_embed + stage2_embed
            x = self.norm(x)

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": None,
            "alibi_bias": None,
            "alibi_scale": None,
            "encoder_mask": mask_info,
        }

    def convmae_random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
        #        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = ids_shuffle.argsort(dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        #        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return MaskInfo(
            x_unmasked=None,
            mask=mask,
            ids_restore=ids_restore.unsqueeze(-1).expand(-1, -1, 768),
            ids_keep=ids_keep,
        )

    def remove_pretraining_modules(self, keep_decoder=False):
        super().remove_pretraining_modules(keep_decoder=keep_decoder)
        if not keep_decoder:
            self.norm = None
        self.stage1_output_decode = self.stage2_output_decode = None
