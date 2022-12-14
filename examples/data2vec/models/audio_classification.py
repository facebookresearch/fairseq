# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import TransposeLast
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)


@dataclass
class AudioClassificationConfig(FairseqDataclass):
    model_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    d2v_args: Any = None
    offload_activations: bool = field(
        default=False, metadata={"help": "offload_activations"}
    )
    min_params_to_wrap: int = field(
        default=int(1e8),
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )
    ddp_backend: str = II("distributed_training.ddp_backend")

    prediction_mode: str = "lin_softmax"
    eval_prediction_mode: Optional[str] = None
    conv_kernel: int = -1
    conv_stride: int = 1
    two_convs: bool = False
    extreme_factor: float = 1.0

    conv_feature_layers: Optional[str] = field(
        default=None,
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )

    mixup_prob: float = 1.0
    source_mixup: float = -1
    same_mixup: bool = True
    label_mixup: bool = False

    gain_mode: str = "none"


@register_model("audio_classification", dataclass=AudioClassificationConfig)
class AudioClassificationModel(BaseFairseqModel):
    def __init__(self, cfg: AudioClassificationConfig, num_classes):
        super().__init__()

        self.apply_mask = cfg.apply_mask
        self.cfg = cfg

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "mask_dropout": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            "mixup": -1,
        }

        if cfg.conv_feature_layers is not None:
            arg_overrides["conv_feature_layers"] = cfg.conv_feature_layers

        if cfg.d2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.model_path, arg_overrides
            )
            d2v_args = state.get("cfg", None)
            if d2v_args is None:
                d2v_args = convert_namespace_to_omegaconf(state["args"])
            d2v_args.criterion = None
            d2v_args.lr_scheduler = None
            cfg.d2v_args = d2v_args

            logger.info(d2v_args)

        else:
            state = None
            d2v_args = cfg.d2v_args

        model_normalized = d2v_args.task.get(
            "normalize", d2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(d2v_args):
                d2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        d2v_args.task.data = cfg.data
        task = tasks.setup_task(d2v_args.task)
        model = task.build_model(d2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        d = d2v_args.model.encoder_embed_dim

        self.d2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        for p in self.parameters():
            p.param_group = "pretrained"

        if cfg.prediction_mode == "proj_avg_proj":
            self.proj = nn.Linear(d, d * 2)
            self.proj2 = nn.Linear(d * 2, num_classes)

            for p in self.proj.parameters():
                p.param_group = "projection"
            for p in self.proj2.parameters():
                p.param_group = "projection"
        elif self.cfg.prediction_mode == "summary_proj":
            self.proj = nn.Linear(d // 3, num_classes)
            for p in self.proj.parameters():
                p.param_group = "projection"
        elif self.cfg.conv_kernel > 1 and not self.cfg.two_convs:
            self.proj = nn.Sequential(
                TransposeLast(),
                nn.Conv1d(d, num_classes, kernel_size=self.cfg.conv_kernel, stride=self.cfg.conv_stride),
                TransposeLast(),
            )
            for p in self.proj.parameters():
                p.param_group = "projection"
        elif self.cfg.conv_kernel > 0 and self.cfg.two_convs:
            self.proj = nn.Sequential(
                TransposeLast(),
                nn.Conv1d(d, d, kernel_size=self.cfg.conv_kernel, stride=self.cfg.conv_stride),
                TransposeLast(),
                nn.GELU(),
                nn.Linear(d, num_classes),
            )
            for p in self.proj.parameters():
                p.param_group = "projection"
        else:
            self.proj = nn.Linear(d, num_classes)
            for p in self.proj.parameters():
                p.param_group = "projection"

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: AudioClassificationConfig, task: FairseqTask):
        """Build a new model instance."""

        assert hasattr(task, "labels"), f"Task {task} must have an attribute 'labels'"

        return cls(cfg, len(task.labels))

    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            model.load_state_dict(state["model"], strict=False)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def compute_gain(self, sound, fs=16_000, min_db=-80.0, mode="A_weighting"):
        if fs == 16000:
            n_fft = 2048
        elif fs == 44100:
            n_fft = 4096
        else:
            raise Exception("Invalid fs {}".format(fs))
        stride = n_fft // 2

        def a_weight(fs, n_fft, min_db=-80.0):
            freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
            freq_sq = np.power(freq, 2)
            freq_sq[0] = 1.0
            weight = 2.0 + 20.0 * (
                2 * np.log10(12194)
                + 2 * np.log10(freq_sq)
                - np.log10(freq_sq + 12194 ** 2)
                - np.log10(freq_sq + 20.6 ** 2)
                - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                - 0.5 * np.log10(freq_sq + 737.9 ** 2)
            )
            weight = np.maximum(weight, min_db)

            return weight

        gain = []
        for i in range(0, len(sound) - n_fft + 1, stride):
            if mode == "RMSE":
                g = np.mean(sound[i : i + n_fft] ** 2)
            elif mode == "A_weighting":
                spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i : i + n_fft])
                power_spec = np.abs(spec) ** 2
                a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
                g = np.sum(a_weighted_spec)
            else:
                raise Exception("Invalid mode {}".format(mode))
            gain.append(g)

        gain = np.array(gain)
        gain = np.maximum(gain, np.power(10, min_db / 10))
        gain_db = 10 * np.log10(gain)

        return gain_db

    # adapted from https://github.com/mil-tokyo/bc_learning_sound/blob/master/utils.py
    def compute_gain_torch(self, sound, fs=16_000, min_db=-80.0, mode="A_weighting"):
        if fs == 16000:
            n_fft = 2048
        elif fs == 44100:
            n_fft = 4096
        else:
            raise Exception("Invalid fs {}".format(fs))

        if mode == "A_weighting":
            if not hasattr(self, f"a_weight"):
                self.a_weight = {}

            if fs not in self.a_weight:

                def a_weight(fs, n_fft, min_db=-80.0):
                    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
                    freq_sq = freq ** 2
                    freq_sq[0] = 1.0
                    weight = 2.0 + 20.0 * (
                        2 * np.log10(12194)
                        + 2 * np.log10(freq_sq)
                        - np.log10(freq_sq + 12194 ** 2)
                        - np.log10(freq_sq + 20.6 ** 2)
                        - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                        - 0.5 * np.log10(freq_sq + 737.9 ** 2)
                    )
                    weight = np.maximum(weight, min_db)

                    return weight

                self.a_weight[fs] = torch.from_numpy(
                    np.power(10, a_weight(fs, n_fft, min_db) / 10)
                ).to(device=sound.device)

        sound = sound.unfold(-1, n_fft, n_fft // 2)

        if mode == "RMSE":
            sound = sound ** 2
            g = sound.mean(-1)
        elif mode == "A_weighting":
            w = torch.hann_window(n_fft, device=sound.device) * sound
            spec = torch.fft.rfft(w)
            power_spec = spec.abs() ** 2
            a_weighted_spec = power_spec * self.a_weight[fs]
            g = a_weighted_spec.sum(-1)
        else:
            raise Exception("Invalid mode {}".format(mode))

        gain = torch.maximum(g, torch.tensor(10 ** (min_db / 10), device=g.device))
        gain_db = 10 * torch.log10(gain)

        return gain_db

    def forward(self, source, padding_mask, label=None, **kwargs):

        if self.cfg.source_mixup >= 0 and self.training and self.cfg.mixup_prob > 0:
            with torch.no_grad():
                mixed_source = source
                mix_mask = None
                if self.cfg.mixup_prob < 1:
                    mix_mask = (
                        torch.empty((source.size(0),), device=source.device)
                        .bernoulli_(self.cfg.mixup_prob)
                        .bool()
                    )
                    mixed_source = source[mix_mask]

                r = (
                    torch.FloatTensor(
                        1 if self.cfg.same_mixup else mixed_source.size(0)
                    )
                    .uniform_(max(1e-6, self.cfg.source_mixup), 1)
                    .to(dtype=source.dtype, device=source.device)
                )

                mixup_perm = torch.randperm(source.size(0))
                s2 = source[mixup_perm]

                if self.cfg.gain_mode == "none":
                    p = r.unsqueeze(-1)
                    if mix_mask is not None:
                        s2 = s2[mix_mask]
                else:
                    if self.cfg.gain_mode == "naive_rms":
                        G1 = source.pow(2).mean(dim=-1).sqrt()
                    else:
                        G1, _ = self.compute_gain_torch(
                            source, mode=self.cfg.gain_mode
                        ).max(-1)
                        G1 = G1.to(dtype=source.dtype)

                    G2 = G1[mixup_perm]

                    if mix_mask is not None:
                        G1 = G1[mix_mask]
                        G2 = G2[mix_mask]
                        s2 = s2[mix_mask]

                    p = 1 / (1 + 10 ** ((G1 - G2) / 20) * (1 - r) / r)
                    p = p.unsqueeze(-1)

                mixed = (p * mixed_source) + (1 - p) * s2

                if mix_mask is None:
                    source = mixed / torch.sqrt(p ** 2 + (1 - p) ** 2)
                else:
                    source[mix_mask] = mixed / torch.sqrt(p ** 2 + (1 - p) ** 2)

                if label is not None and self.cfg.label_mixup:
                    r = r.unsqueeze(-1)
                    if mix_mask is None:
                        label = label * r + (1 - r) * label[mixup_perm]
                    else:
                        label[mix_mask] = (
                            label[mix_mask] * r + (1 - r) * label[mixup_perm][mix_mask]
                        )

        d2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.d2v_model.extract_features(**d2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]
            if padding_mask is not None:
                x[padding_mask] = 0

        x = self.final_dropout(x)

        if self.training or (
            self.cfg.eval_prediction_mode is None or self.cfg.eval_prediction_mode == ""
        ):
            prediction_mode = self.cfg.prediction_mode
        else:
            prediction_mode = self.cfg.eval_prediction_mode

        if prediction_mode == "average_before":
            x = x.mean(dim=1)

        if prediction_mode != "summary_mha" and prediction_mode != "summary_proj" and prediction_mode != "cls":
            x = self.proj(x)

        logits = True
        if prediction_mode == "lin_softmax":
            x = F.logsigmoid(x.float())
            x = torch.logsumexp(x + x, dim=1) - torch.logsumexp(x, dim=1)
            x = x.clamp(max=0)
            x = x - torch.log(-(torch.expm1(x)))
        elif prediction_mode == "extremized_odds":
            x = x.float().sum(dim=1)
            x = x * self.cfg.extreme_factor
        elif prediction_mode == "average_before":
            x = x.float()
        elif prediction_mode == "average":
            x = x.float().mean(dim=1)
        elif prediction_mode == "average_sigmoid":
            x = torch.sigmoid(x.float())
            x = x.mean(dim=1)
            logits = False
        elif prediction_mode == "max":
            x, _ = x.float().max(dim=1)
        elif prediction_mode == "max_sigmoid":
            x = torch.sigmoid(x.float())
            x, _ = x.float().max(dim=1)
            logits = False
        elif prediction_mode == "proj_avg_proj":
            x = x.mean(dim=1)
            x = self.proj2(x)
        elif prediction_mode == "summary_mha" or prediction_mode == "summary_proj":
            x = self.d2v_model.summary(
                x, padding_mask, proj=prediction_mode == "summary_proj"
            )
            x = x.type_as(source)
            x = self.proj(x)
        elif prediction_mode == "cls":
            x = x[:,0]
            x = self.proj(x)
        else:
            raise Exception(f"unknown prediction mode {prediction_mode}")

        if label is None:
            return torch.sigmoid(x) if logits else x

        x = torch.nan_to_num(x)

        if logits:
            loss = F.binary_cross_entropy_with_logits(
                x, label.float(), reduction="none"
            )
        else:
            loss = F.binary_cross_entropy(x, label.float(), reduction="none")

        result = {
            "losses": {
                "main": loss,
            },
            "sample_size": label.sum(),
        }

        if not self.training:
            result["_predictions"] = torch.sigmoid(x) if logits else x
            result["_targets"] = label

        return result
