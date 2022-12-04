# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
import numpy as np

from omegaconf import II

import torch

from fairseq.models import register_model

from ..tasks.multimodal import Modality

from .multi.base import (
    MaskSeed,
    get_annealed_rate,
)
from .data2vec_multi import Data2VecMultiConfig, Data2VecMultiModel

logger = logging.getLogger(__name__)


@dataclass
class Data2VecMulti2EmaConfig(Data2VecMultiConfig):

    slow_ema_decay: float = field(
        default=0.99995, metadata={"help": "initial ema decay rate"}
    )
    slow_ema_end_decay: float = field(
        default=0.99999, metadata={"help": "final ema decay rate"}
    )
    slow_ema_anneal_end_step: int = II("optimization.max_update")
    slow_ema_weight: float = 1


@register_model("data2vec_multi_2ema", dataclass=Data2VecMulti2EmaConfig)
class Data2VecMulti2EmaModel(Data2VecMultiModel):

    cfg: Data2VecMulti2EmaConfig

    def __init__(
        self, cfg: Data2VecMulti2EmaConfig, modalities, skip_ema=False, task=None
    ):
        super().__init__(cfg, modalities, skip_ema, task)

        if not skip_ema:
            self.slow_ema = self.make_ema_teacher(cfg.slow_ema_decay)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.slow_ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.slow_ema is not None:
            if self.cfg.slow_ema_decay != self.cfg.slow_ema_end_decay:
                if num_updates >= self.cfg.slow_ema_anneal_end_step:
                    decay = self.cfg.slow_ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.slow_ema_decay,
                        self.cfg.slow_ema_end_decay,
                        num_updates,
                        self.cfg.slow_ema_anneal_end_step,
                    )
                self.slow_ema.set_decay(decay)
            if self.slow_ema.get_decay() < 1:
                self.slow_ema.step(self.blocks if self.cfg.ema_encoder_only else self)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.slow_ema is not None:
            state[prefix + "_slow_ema"] = self.slow_ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_slow_ema"
        if self.slow_ema is not None:
            assert k in state_dict
            self.slow_ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecMulti2EmaConfig, task=None):
        """Build a new model instance."""
        if task is None or not hasattr(task, "supported_modalities"):
            modalities = (
                [cfg.supported_modality]
                if cfg.supported_modality is not None
                else [
                    Modality.AUDIO,
                    Modality.IMAGE,
                    Modality.TEXT,
                ]
            )
        else:
            modalities = task.supported_modalities
        return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)

    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
    ):
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality

        if isinstance(mode, Modality):
            mode = mode.name

        feature_extractor = self.modality_encoders[mode]

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

        xs = []

        if self.shared_decoder is not None:
            dx = self.forward_decoder(
                x,
                feature_extractor,
                self.shared_decoder,
                encoder_mask,
            )
            xs.append(dx)
        if feature_extractor.decoder is not None:
            dx = self.forward_decoder(
                x,
                feature_extractor,
                feature_extractor.decoder,
                encoder_mask,
            )
            xs.append(dx)
            orig_x = x

        assert len(xs) > 0

        p = next(self.ema.model.parameters())
        device = x.device
        dtype = x.dtype
        ema_device = p.device
        ema_dtype = p.dtype

        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        if ema_device != device or ema_dtype != dtype:
            logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
            self.ema.model = self.ema.model.to(dtype=dtype, device=device)
            self.slow_ema.model = self.slow_ema.model.to(dtype=dtype, device=device)
            ema_dtype = dtype

            def to_device(d):
                for k, p in d.items():
                    if isinstance(d[k], dict):
                        to_device(d[k])
                    else:
                        d[k] = p.to(device=device)

            to_device(self.ema.fp32_params)
            to_device(self.slow_ema.fp32_params)

        targets = []
        masked = encoder_mask.mask.unsqueeze(-1)
        masked_b = encoder_mask.mask.bool()
        sample_size = masked.sum().long() if not self.mean_loss else 1

        if xs[0].size(1) == masked_b.size(1):
            xs = [x[masked_b] for x in xs]
        else:
            xs = [x.reshape(-1, x.size(-1)) for x in xs]

        for tm in [self.slow_ema.model, self.ema.model]:
            with torch.no_grad():
                tm.eval()

                if self.cfg.ema_encoder_only:
                    assert target is None
                    ema_input = extractor_out["local_features"]
                    ema_input = feature_extractor.contextualized_features(
                        ema_input.to(dtype=ema_dtype),
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )
                    ema_blocks = tm
                else:
                    ema_blocks = tm.blocks
                    if feature_extractor.modality_cfg.ema_local_encoder:
                        inp = (
                            target.to(dtype=ema_dtype)
                            if target is not None
                            else source.to(dtype=ema_dtype)
                        )
                        ema_input = tm.modality_encoders[mode](
                            inp,
                            padding_mask,
                            mask=False,
                            remove_masked=False,
                        )
                    else:
                        assert target is None
                        ema_input = extractor_out["local_features"]
                        ema_feature_enc = tm.modality_encoders[mode]
                        ema_input = ema_feature_enc.contextualized_features(
                            ema_input.to(dtype=ema_dtype),
                            padding_mask,
                            mask=False,
                            remove_masked=False,
                        )

                ema_padding_mask = ema_input["padding_mask"]
                ema_alibi_bias = ema_input.get("alibi_bias", None)
                ema_alibi_scale = ema_input.get("alibi_scale", None)
                ema_input = ema_input["x"]

                y = []
                ema_x = []
                extra_tokens = feature_extractor.modality_cfg.num_extra_tokens
                for i, blk in enumerate(ema_blocks):
                    ab = ema_alibi_bias
                    if ab is not None and alibi_scale is not None:
                        scale = (
                            ema_alibi_scale[i]
                            if ema_alibi_scale.size(0) > 1
                            else ema_alibi_scale.squeeze(0)
                        )
                        ab = ab * scale.type_as(ab)

                    ema_input, lr = blk(
                        ema_input,
                        padding_mask=ema_padding_mask,
                        alibi_bias=ab,
                    )
                    y.append(lr[:, extra_tokens:])
                    ema_x.append(ema_input[:, extra_tokens:])

            y = self.make_targets(y, self.average_top_k_layers)
            orig_targets = y

            if self.cfg.clone_batch > 1:
                y = y.repeat_interleave(self.cfg.clone_batch, 0)

            targets.append(y[masked_b])

        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        sample_size = result["sample_size"]

        if self.cfg.cls_loss > 0:
            assert extra_tokens > 0
            cls_target = orig_targets.mean(dim=1)
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
            cls_pred = x[:, extra_tokens - 1]
            result["losses"]["cls"] = self.d2v_loss(cls_pred, cls_target) * (
                self.cfg.cls_loss * sample_size
            )

        if self.cfg.recon_loss > 0:

            with torch.no_grad():
                target = feature_extractor.patchify(source)
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

                if self.cfg.clone_batch > 1:
                    target = target.repeat_interleave(self.cfg.clone_batch, 0)

                if masked_b is not None:
                    target = target[masked_b]

            recon = xs[0]
            if self.recon_proj is not None:
                recon = self.recon_proj(recon)

            result["losses"]["recon"] = (
                self.d2v_loss(recon, target.float()) * self.cfg.recon_loss
            )

        if self.cfg.d2v_loss > 0:
            for i, x in enumerate(xs):
                for j, y in enumerate(targets):
                    mult = self.cfg.slow_ema_weight if j == 0 else self.cfg.d2v_loss
                    reg_loss = self.d2v_loss(x, y)
                    n = (
                        f"{mode}_regression_{i}_{j}"
                        if len(xs) > 1
                        else f"{mode}_regression_{j}"
                    )
                    result["losses"][n] = reg_loss * mult

        suffix = "" if len(self.modalities) == 1 else f"_{mode}"
        with torch.no_grad():
            if encoder_mask is not None:
                result["masked_pct"] = 1 - (
                    encoder_mask.ids_keep.size(1) / encoder_mask.ids_restore.size(1)
                )
            for i, x in enumerate(xs):
                n = f"pred_var{suffix}_{i}" if len(xs) > 1 else f"pred_var{suffix}"
                result[n] = self.compute_var(x.float())
            if self.ema is not None:
                for k, v in self.ema.logs.items():
                    result[k] = v

            y = y.float()
            result[f"target_var{suffix}"] = self.compute_var(y)

            if self.num_updates > 5000:
                if result[f"target_var{suffix}"] < self.cfg.min_target_var:
                    logger.error(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )
                    raise Exception(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )

                for k in result.keys():
                    if k.startswith("pred_var") and result[k] < self.cfg.min_pred_var:
                        logger.error(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )
                        raise Exception(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )

            result["ema_decay"] = self.ema.get_decay() * 1000
            result["slow_ema_decay"] = self.slow_ema.get_decay() * 1000

        return result

    def remove_pretraining_modules(self, modality=None, keep_decoder=False):
        super().remove_pretraining_modules(modality=modality, keep_decoder=keep_decoder)
        self.slow_ema = None
