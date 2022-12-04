# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import math
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, List

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    MlpFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
    PosEmb,
)
from fairseq.modules import (
    GradMultiply,
    Fp32LayerNorm,
    LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.utils import index_put


logger = logging.getLogger(__name__)


@dataclass
class Data2VecAudioConfig(Wav2Vec2Config):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    use_l1: bool = False
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )
    average_layer_list: Optional[List[int]] = None
    target_layerdrop: float = 0

    layer_to_layer: bool = False
    layer_to_layer_reverse: bool = False
    layer_to_layer_share_dec: bool = False
    layer_to_layer_share_proj: bool = False
    layer_to_layer_dec_only: bool = False

    clone_batch: int = 1

    max_layer: bool = False

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    normalize_target_magnitude: float = 0
    clamp_target_magnitude: float = 0
    tanh_targets_div: float = 0
    tanh_targets_mult: float = 1
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    end_of_block_targets: bool = False

    reset_target_every: int = 0
    reset_target_every_factor: float = 1
    reset_warmup_pct: float = 0.05

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})

    adaptive_ema_decay_mom: float = -1
    adaptive_ema_decay_mom2: float = -1
    adaptive_ema_decay_step_up: float = 1e-3
    adaptive_ema_decay_step_down: float = 1e-5
    adaptive_after_anneal: bool = False

    clamp_to_multiple: float = 0
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    max_sample_size: int = II("task.max_sample_size")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    affine_features_norm: bool = True

    layer_norm_pred: bool = False

    debug: bool = False

    mask_noise_std: float = 0

    no_final_proj: bool = False

    mae_style: bool = False
    mae_decoder_dim: int = 512
    mae_decoder_layers: int = 6
    mae_decoder_heads: int = 16
    mae_decoder_dropout: float = 0
    mae_ffn_factor: int = 4
    mae_decoder_pos_type: PosEmb = PosEmb.CONV

    mae_use_in_proj: bool = False
    mae_use_in_proj_ln: bool = False
    mae_use_in_proj_act: bool = False
    mae_use_in_proj_residual: bool = False

    mae_use_out_proj: bool = False
    mae_mask_pos: bool = False
    mae_zero_pos: bool = False
    mae_share_pos_enc: bool = False

    mae_conv_decoder_layers: int = 0
    mae_conv_decoder_no_ln: bool = False
    mae_conv_decoder_instance_norm: bool = False
    mae_conv_decoder_batch_norm: bool = False
    mae_conv_decoder_no_act: bool = False
    mae_conv_decoder_kernel: int = 6
    mae_conv_decoder_groups: int = 1
    mae_conv_decoder_residual: bool = False
    mae_conv_decoder_residual_context_only: bool = False
    mae_conv_residual_avg: bool = False
    mae_conv_residual_act: bool = False
    mae_conv_residual_ln: bool = False

    mae_conv_decoder_ffn: bool = False
    mae_conv_decoder_last_ln: bool = False

    order_weight: float = 1.0

    mae_decoder_zero_mask: bool = False

    mae_in_proj_is_decoder: bool = False
    mae_final_proj_is_decoder: bool = False
    mae_final_proj_separate: bool = False
    mae_encoder_decoder_attn: bool = False
    mae_mask_pct_future: float = 0
    mae_no_self_attn: bool = False

    mae_no_ffn: bool = False
    mae_no_norm1: bool = False
    mae_no_norm2: bool = False
    mae_no_prenorm: bool = False
    mae_no_enc_dec_resid: bool = False

    predict_all: bool = False
    zero_mask: bool = False

    use_alibi_encoder: bool = False
    use_alibi_decoder: bool = False

    ema_cosine_decay: bool = False
    ema_exponential_decay: bool = False
    ema_exponential_decay_gamma: float = 0.05
    ema_warmup_updates: int = 8000

    ema_train: bool = False

    mask_prob_start: float = II("model.mask_prob")
    mask_warmup: int = 0

    seed: int = II("common.seed")


def get_annealed_rate(start, end, curr_step, total_steps):
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


def get_cosine_rate(num_updates, warmup_updates, max_update, max_lr=1, min_lr=0):
    if num_updates < warmup_updates:
        return num_updates / warmup_updates
    else:
        curr_updates = num_updates - warmup_updates
        period = max_update - warmup_updates
        i = math.floor(curr_updates / period)
        t_i = period
        t_curr = curr_updates - (period * i)

        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))


def get_exponential_rate(num_updates, warmup_updates, max_update, gamma=0.05):
    if num_updates < warmup_updates:
        return num_updates / warmup_updates
    else:
        curr_updates = num_updates - warmup_updates
        period = max_update - warmup_updates

        return gamma ** (curr_updates / period)


def get_alibi(
    max_positions: int,
    attention_heads: int,
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))
    # prepare alibi position linear bias. Note that wav2vec2 is non
    # autoregressive model so we want a symmetric mask with 0 on the
    # diagonal and other wise linear decreasing valuees
    pos_bias = (
        torch.abs(torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1))
        * -1
    )
    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )
    return alibi_bias


@torch.no_grad()
def get_alibi_bias(alibi_biases, heads, batch_size, time_steps, dtype, device):
    buffered = alibi_biases.get(heads, None)

    target_size = heads * batch_size
    if (
        buffered is None
        or buffered.size(0) < target_size
        or buffered.size(1) < time_steps
        or buffered.dtype != dtype
        or buffered.device != device
    ):
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads

        buffered = get_alibi(bt, heads).to(dtype=dtype, device=device).repeat(bn, 1, 1)

        alibi_biases[heads] = buffered

    return buffered[:target_size, :time_steps, :time_steps]


def masked_alibi(alibi_bias, mask_indices, orig_B, orig_T):
    alibi_bias = alibi_bias.view(orig_B, -1, orig_T, orig_T)
    H = alibi_bias.size(1)
    alibi_mask = mask_indices.unsqueeze(1)
    alibi_bias = alibi_bias.masked_select(alibi_mask.unsqueeze(-1))
    alibi_bias = alibi_bias.view(orig_B, H, -1, orig_T)
    M = alibi_bias.size(-2)
    alibi_bias = alibi_bias.masked_select(alibi_mask.unsqueeze(-2))
    alibi_bias = alibi_bias.view(-1, M, M)
    return alibi_bias


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


@register_model("data2vec_audio", dataclass=Data2VecAudioConfig)
class Data2VecAudioModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecAudioConfig, skip_ema=False):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.extractor_embed = feature_enc_layers[-1][0]

        self.ema = None
        self.target_model = None
        self.grad_mult_stack = []
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.average_layer_list = cfg.average_layer_list
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.alibi_biases = {}

        if cfg.mlp_encoder:
            self.feature_extractor = MlpFeatureExtractionModel(
                n_in=320,
                n_out=self.extractor_embed,
                n_layers=cfg.mlp_layers,
                layer_norm=cfg.mlp_layernorm,
            )
        else:
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
                affine_norms=cfg.affine_extractor_norm,
                fsdp_fp16=cfg.ddp_backend == "fully_sharded" and cfg.fp16,
            )

        self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim)

        self.mask_prob = cfg.mask_prob_start
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.clone_hash = [
            int(hash((self.cfg.seed, ind)) % 1e10)
            for ind in range(self.cfg.clone_batch - 1)
        ]
        self.clone_hash = torch.tensor([0] + self.clone_hash).long().view(1, -1)

        mask_dim = (
            cfg.encoder_embed_dim
            if not cfg.mae_style
            or not cfg.mae_encoder_decoder_attn
            or cfg.mae_conv_decoder_layers > 0
            else cfg.mae_decoder_dim
        )

        if cfg.mae_style and cfg.mae_mask_pos:
            self.encoder_mask_emb = (
                nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())
                if cfg.mask_noise_std <= 0
                else None
            )

        self.mask_emb = (
            nn.Parameter(torch.FloatTensor(mask_dim).uniform_())
            if cfg.mask_noise_std <= 0
            else None
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(
            self.extractor_embed, elementwise_affine=cfg.affine_features_norm
        )

        if cfg.ema_decay > 0:
            if not skip_ema:
                self.make_ema_teacher()
        else:
            self.target_model = self.make_target_model()

        tdim = self.embed

        if cfg.mae_use_out_proj or not cfg.mae_style:
            in_dim = self.embed
        else:
            in_dim = cfg.mae_decoder_dim

        if cfg.layer_to_layer and not cfg.layer_to_layer_share_proj:
            self.final_proj = nn.ModuleList(
                [nn.Linear(in_dim, tdim) for _ in range(self.average_top_k_layers)]
            )
        elif cfg.layer_norm_pred:
            self.final_proj = nn.Sequential(
                nn.Linear(in_dim, tdim), Fp32LayerNorm(tdim)
            )
        else:
            if self.cfg.no_final_proj:
                assert in_dim == tdim
                self.final_proj = nn.Identity()
            else:
                self.final_proj = nn.Linear(in_dim, tdim)

        self.decoder = None
        self.decoder_in_proj = None
        self.decoder_out_proj = None
        if cfg.mae_style:
            t_cfg = copy.deepcopy(cfg)
            t_cfg.encoder_layers = cfg.mae_decoder_layers
            t_cfg.encoder_embed_dim = cfg.mae_decoder_dim
            t_cfg.encoder_ffn_embed_dim = cfg.mae_decoder_dim * cfg.mae_ffn_factor
            t_cfg.encoder_attention_heads = cfg.mae_decoder_heads
            t_cfg.pos_emb_type = (
                cfg.mae_decoder_pos_type if not cfg.mae_share_pos_enc else PosEmb.NONE
            )
            t_cfg.encoder_layerdrop = 0

            if cfg.mae_conv_decoder_layers > 0:
                if (
                    cfg.layer_to_layer
                    and cfg.layer_to_layer_dec_only
                    and cfg.layer_to_layer_share_dec
                ):
                    assert cfg.mae_conv_decoder_layers == cfg.average_top_k_layers

                def make_block(in_dim):
                    block = [
                        nn.Conv1d(
                            in_dim,
                            cfg.mae_decoder_dim,
                            kernel_size=cfg.mae_conv_decoder_kernel,
                            padding=cfg.mae_conv_decoder_kernel // 2,
                            groups=cfg.mae_conv_decoder_groups,
                        ),
                        SamePad(cfg.mae_conv_decoder_kernel),
                    ]

                    if cfg.mae_conv_decoder_batch_norm:
                        block.append(nn.BatchNorm1d(cfg.mae_decoder_dim))
                    elif cfg.mae_conv_decoder_instance_norm:
                        block.append(nn.InstanceNorm1d(cfg.mae_decoder_dim))
                    elif not cfg.mae_conv_decoder_no_ln:
                        block.extend(
                            [
                                TransposeLast(),
                                LayerNorm(
                                    cfg.mae_decoder_dim, elementwise_affine=False
                                ),
                                TransposeLast(),
                            ]
                        )

                    if not cfg.mae_conv_decoder_no_act:
                        block.append(nn.GELU())
                    return nn.Sequential(*block)

                blocks = [
                    make_block(
                        cfg.encoder_embed_dim
                        if i == 0 and not self.cfg.mae_use_in_proj
                        else cfg.mae_decoder_dim
                    )
                    for i in range(cfg.mae_conv_decoder_layers)
                ]

                if cfg.layer_to_layer and not cfg.layer_to_layer_share_dec:
                    self.decoder = nn.ModuleList(
                        [
                            nn.Sequential(*blocks)
                            for _ in range(cfg.average_top_k_layers)
                        ]
                    )
                else:
                    self.decoder = nn.Sequential(*blocks)

                self.post_decoder = None
                if self.cfg.mae_conv_decoder_ffn:
                    self.post_decoder = nn.Sequential(
                        nn.Linear(cfg.mae_decoder_dim, cfg.mae_decoder_dim * 4),
                        nn.GELU(),
                        nn.Linear(cfg.mae_decoder_dim * 4, cfg.mae_decoder_dim),
                    )
                self.post_decoder_ln = None
                if self.cfg.mae_conv_decoder_last_ln:
                    self.post_decoder_ln = LayerNorm(cfg.mae_decoder_dim)

                if cfg.mae_use_in_proj:
                    assert not cfg.layer_to_layer
                    proj_block = [
                        nn.Conv1d(
                            cfg.encoder_embed_dim,
                            cfg.mae_decoder_dim,
                            kernel_size=1,
                        )
                    ]

                    if cfg.mae_use_in_proj_ln:
                        proj_block.extend(
                            [
                                TransposeLast(),
                                LayerNorm(cfg.mae_decoder_dim),
                                TransposeLast(),
                            ]
                        )

                    if cfg.mae_use_in_proj_act:
                        proj_block.append(nn.GELU())

                    self.decoder_in_proj = nn.Sequential(*proj_block)

            else:
                self.decoder = TransformerEncoder(
                    t_cfg,
                    cfg.encoder_embed_dim if cfg.mae_encoder_decoder_attn else None,
                    not cfg.mae_no_self_attn,
                    not cfg.mae_no_ffn,
                    not cfg.mae_no_norm1,
                    not cfg.mae_no_norm2,
                    not cfg.mae_no_prenorm,
                    not cfg.mae_no_enc_dec_resid,
                )

            if (
                (
                    not self.cfg.mae_encoder_decoder_attn
                    or t_cfg.pos_emb_type == PosEmb.CONV
                )
                and cfg.encoder_embed_dim != cfg.mae_decoder_dim
                and cfg.mae_conv_decoder_layers == 0
            ):
                self.decoder_in_proj = nn.Linear(
                    cfg.encoder_embed_dim, cfg.mae_decoder_dim
                )

            if cfg.mae_use_out_proj:
                self.decoder_out_proj = nn.Linear(
                    cfg.mae_decoder_dim, cfg.encoder_embed_dim
                )

            for p in self.parameters():
                p.param_group = "encoder"
            for p in self.decoder.parameters():
                p.param_group = "decoder"

            if self.cfg.mae_in_proj_is_decoder:
                if self.decoder_in_proj is not None:
                    for p in self.decoder_in_proj.parameters():
                        p.param_group = "decoder"
                if self.decoder_out_proj is not None:
                    for p in self.decoder_out_proj.parameters():
                        p.param_group = "decoder"

            if self.cfg.mae_final_proj_separate:
                for p in self.final_proj.parameters():
                    p.param_group = "final"
            elif self.cfg.mae_final_proj_is_decoder:
                for p in self.final_proj.parameters():
                    p.param_group = "decoder"

        self.prototypes = None
        if cfg.adaptive_ema_decay_mom >= 0:
            self.register_buffer(
                "decay", torch.tensor(self.cfg.ema_decay, dtype=torch.float32)
            )
            self.register_buffer("smooth_loss", torch.tensor(0, dtype=torch.float32))
            self.register_buffer("smooth_loss2", torch.tensor(0, dtype=torch.float32))

        self.num_updates = 0

    @torch.no_grad()
    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
            log_norms=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        model_copy = self.make_target_model()

        self.ema = EMAModule(
            model_copy,
            ema_config,
            copy_model=False,
            skip_keys=skip_keys,
        )

    def make_target_model(self):
        logger.info("making target model")
        if self.cfg.ema_transformer_only:
            model_copy = TransformerEncoder(self.cfg)
            for p_s, p_t in zip(self.encoder.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            model_copy = Data2VecAudioModel(self.cfg, skip_ema=True)
            for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        model_copy.requires_grad_(False)
        return model_copy

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        self.mask_prob = get_annealed_rate(
            self.cfg.mask_prob_start,
            self.cfg.mask_prob,
            num_updates,
            self.cfg.mask_warmup,
        )

        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates == num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            if self.cfg.ema_cosine_decay:
                mult = get_cosine_rate(
                    num_updates, self.cfg.ema_warmup_updates, self.cfg.max_update
                )
                ema_lr = 1 - self.cfg.ema_decay
                ema_lr *= mult
                decay = 1 - ema_lr
                self.ema.set_decay(decay)
            elif self.cfg.ema_exponential_decay:
                mult = get_exponential_rate(
                    num_updates,
                    self.cfg.ema_warmup_updates,
                    self.cfg.max_update,
                    self.cfg.ema_exponential_decay_gamma,
                )
                ema_lr = 1 - self.cfg.ema_decay
                ema_lr *= mult
                decay = 1 - ema_lr
                self.ema.set_decay(decay)
            elif self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                    if self.cfg.adaptive_ema_decay_mom >= 0:
                        self.decay = torch.full_like(
                            self.decay, fill_value=decay
                        ).float()
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)
        elif self.training and self.target_model is not None:
            if not hasattr(self, "reset_target_steps"):
                self.reset_target_steps = dict()
                if self.cfg.reset_target_every > 0:
                    upd = period = self.cfg.reset_target_every
                    prev = 0
                    while upd < self.cfg.max_update:
                        warmup_steps = int((upd - prev) * self.cfg.reset_warmup_pct)
                        prev = upd
                        self.reset_target_steps[upd] = warmup_steps
                        period *= self.cfg.reset_target_every_factor
                        upd += int(period)
                    logger.info(f"Resetting updates: {self.reset_target_steps}")
            if num_updates in self.reset_target_steps:
                self.target_model = self.make_target_model()

                p = next(self.target_model.parameters())
                s_p = next(self.parameters())

                if s_p.device != p.device or s_p.dtype != p.dtype:
                    self.target_model = self.target_model.to(
                        dtype=s_p.dtype, device=s_p.device
                    )

                self.grad_mult_stack = np.linspace(
                    0.99, 0.01, self.reset_target_steps[num_updates]
                ).tolist()

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
    def build_model(cls, cfg: Data2VecAudioConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        id,
        mask_indices=None,
        mask_channel_indices=None,
        skip_apply=False,
        mask_prob=None,
        zero_mask=False,
    ):
        B, T, C = x.shape

        if mask_prob is None:
            mask_prob = self.mask_prob

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if mask_prob > 0:
            if mask_indices is None:
                future_mask = self.cfg.mae_mask_pct_future > 0 and self.cfg.mae_style
                if future_mask:
                    right_T = int(T * self.cfg.mae_mask_pct_future)
                    T = T - right_T

                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                    add_masks=self.cfg.add_masks,
                    seed=self.cfg.seed,
                    epoch=self.num_updates,
                    indices=id,
                )
                if future_mask:
                    mask_indices = np.pad(
                        mask_indices, ((0, 0), (0, right_T)), constant_values=1
                    )

                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            if (
                self.mask_emb is not None
                and x.size(-1) == self.mask_emb.size(-1)
                and not zero_mask
            ):
                masks = self.mask_emb
            else:
                masks = x.new_empty((*(x[mask_indices]).shape,)).normal_(
                    0, self.cfg.mask_noise_std
                )
            if not skip_apply:
                x = index_put(x, mask_indices, masks)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            if not skip_apply:
                x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        id=None,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):
        features = source

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        if not self.cfg.mlp_encoder:
            features = features.transpose(1, 2)

        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        pre_encoder_features = None
        if (self.cfg.ema_transformer_only or self.cfg.mae_style) and not features_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
            if self.cfg.clone_batch > 1 and not features_only:
                features = features.repeat_interleave(self.cfg.clone_batch, 0)
                if id is not None:
                    id = id.repeat_interleave(self.cfg.clone_batch, 0)
                    id = id.view(-1, self.cfg.clone_batch) + self.clone_hash.to(id)
                    id = id.view(-1)

            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
                skip_apply=self.cfg.mae_style and not features_only,
                zero_mask=self.cfg.mae_style and features_only and self.cfg.zero_mask,
                id=id,
            )
        else:
            x = features
            mask_indices = None

        orig_B, orig_T = x.size(0), x.size(1)

        encoder_padding_mask = padding_mask

        doing_mae = self.cfg.mae_style and not features_only and mask
        encoder_mask = None
        if doing_mae:
            encoder_mask = mask_indices

            if encoder_mask is not None:
                if self.cfg.mae_zero_pos:
                    x = index_put(x, encoder_mask, 0)
                elif self.cfg.mae_mask_pos:
                    if self.encoder_mask_emb is not None:
                        masks = self.encoder_mask_emb
                    else:
                        masks = x.new_empty((*(x[encoder_mask]).shape,)).normal_(
                            0, self.cfg.mask_noise_std
                        )
                    x = index_put(x, encoder_mask, masks)

            x = self.encoder.add_positions(x)

            B, T = (
                encoder_mask.size(0),
                encoder_mask.size(1) - encoder_mask[0].sum(),
            )
            x = x[~encoder_mask].view(B, T, -1)

            if encoder_padding_mask is not None:
                encoder_padding_mask = encoder_padding_mask[~encoder_mask].view(B, T)

        alibi_bias_enc = None
        if self.cfg.use_alibi_encoder:
            alibi_bias_enc = get_alibi_bias(
                self.alibi_biases,
                self.cfg.encoder_attention_heads,
                batch_size=orig_B,
                time_steps=orig_T,
                dtype=x.dtype,
                device=x.device,
            )

            if encoder_mask is not None:
                alibi_bias_enc = masked_alibi(
                    alibi_bias_enc, (~encoder_mask), orig_B, orig_T
                )

        x, layer_results = self.encoder(
            x,
            padding_mask=encoder_padding_mask,
            layer=layer,
            skip_positions=doing_mae,
            alibi_bias=alibi_bias_enc,
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        if self.decoder is not None:
            if self.cfg.layer_to_layer:
                if self.cfg.layer_to_layer_dec_only:
                    if self.cfg.layer_to_layer_share_dec:
                        _, dec_lr = self.forward_decoder(
                            x,
                            self.decoder,
                            encoder_mask,
                            mask_indices,
                            orig_B,
                            orig_T,
                            padding_mask,
                        )
                        x = [z.transpose(1, 2) for z in dec_lr]
                        if self.cfg.layer_to_layer_reverse:
                            x = list(reversed(x))
                    else:
                        dec_lr = []
                        for dec in self.decoder:
                            xd, _ = self.forward_decoder(
                                x,
                                dec,
                                encoder_mask,
                                mask_indices,
                                orig_B,
                                orig_T,
                                padding_mask,
                            )
                            dec_lr.append(xd)
                        x = dec_lr
                        if self.cfg.layer_to_layer_reverse:
                            x = list(reversed(x))

                elif self.cfg.layer_to_layer_share_dec:
                    dec_lr = []
                    for lr in layer_results[-self.cfg.average_top_k_layers :]:
                        xr = lr[1].clone().transpose(0, 1)
                        xd, _ = self.forward_decoder(
                            xr,
                            self.decoder,
                            encoder_mask,
                            mask_indices,
                            orig_B,
                            orig_T,
                            padding_mask,
                        )
                        dec_lr.append(xd)
                    x = dec_lr
                else:
                    dec_lr = []
                    for lr, dec in zip(
                        layer_results[-self.cfg.average_top_k_layers :], self.decoder
                    ):
                        xr = lr[1].clone().transpose(0, 1)
                        xd, _ = self.forward_decoder(
                            xr,
                            dec,
                            encoder_mask,
                            mask_indices,
                            orig_B,
                            orig_T,
                            padding_mask,
                        )
                        dec_lr.append(xd)
                    x = dec_lr
            else:
                x, _ = self.forward_decoder(
                    x,
                    self.decoder,
                    encoder_mask,
                    mask_indices,
                    orig_B,
                    orig_T,
                    padding_mask,
                )

        tm = None
        if self.ema is not None:
            p = next(self.ema.model.parameters())
            device = features.device
            dtype = features.dtype
            ema_device = p.device
            ema_dtype = p.dtype

            if ema_device != device or ema_dtype != dtype:
                logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
                self.ema.model = self.ema.model.to(dtype=dtype, device=device)

                def to_device(d):
                    for k, p in d.items():
                        if isinstance(d[k], dict):
                            to_device(d[k])
                        else:
                            d[k] = p.to(device=device)

                to_device(self.ema.fp32_params)

            tm = self.ema.model

        elif self.target_model is not None:
            tm = self.target_model

        if tm is not None:
            with torch.no_grad():
                tm.train(mode=self.cfg.ema_train and self.training)

                if self.cfg.ema_transformer_only:
                    ema_input = pre_encoder_features

                    alibi_bias_ema = None
                    if alibi_bias_enc is not None:
                        alibi_bias_ema = get_alibi_bias(
                            self.alibi_biases,
                            self.cfg.encoder_attention_heads,
                            batch_size=ema_input.size(0),
                            time_steps=ema_input.size(1),
                            dtype=ema_input.dtype,
                            device=ema_input.device,
                        )

                    y, y_layer_results = tm.extract_features(
                        ema_input,
                        padding_mask=padding_mask,
                        min_layer=self.cfg.encoder_layers - self.average_top_k_layers
                        if self.average_layer_list is None
                        else 0,
                        alibi_bias=alibi_bias_ema,
                    )
                    y = {
                        "x": y,
                        "padding_mask": padding_mask,
                        "layer_results": y_layer_results,
                    }
                else:
                    assert alibi_bias_enc is None
                    y = tm.extract_features(
                        source=source,
                        padding_mask=orig_padding_mask,
                        mask=False,
                    )

        y = self.make_targets(y, self.average_top_k_layers)

        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)

        if mask_indices is not None and not self.cfg.predict_all:
            y = y[mask_indices]
            if (
                not self.cfg.mae_encoder_decoder_attn
                or self.cfg.mae_conv_decoder_layers > 0
            ):
                if self.cfg.layer_to_layer:
                    x = [dx[mask_indices] for dx in x]
                else:
                    x = x[mask_indices]
            else:
                x = x.reshape(y.size(0), -1)
        else:
            y = y.reshape(-1, y.size(-1))
            x = x.reshape(-1, x.size(-1))

        if self.decoder_out_proj is not None:
            x = self.decoder_out_proj(x)

        if self.cfg.layer_to_layer:
            if self.cfg.layer_to_layer_share_proj:
                x = torch.stack(x, dim=-2)
                x = self.final_proj(x)
            else:
                x = [fp(x_i) for x_i, fp in zip(x, self.final_proj)]
                x = torch.stack(x, dim=-2)
        else:
            x = self.final_proj(x)

        result = {}

        with torch.no_grad():
            result["pred_var"] = self.compute_var(x.float())
            result["max_x"] = x.abs().max()
            if self.ema is not None:
                for k, v in self.ema.logs.items():
                    result[k] = v

        y = y.float()

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["max_y"] = y.abs().max()

        reg_loss, sample_size = self.d2v_loss(x, y)

        result["losses"] = {"regression": reg_loss}

        if "sample_size" not in result:
            result["sample_size"] = sample_size

        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )

        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        if self.cfg.adaptive_ema_decay_mom >= 0:
            with torch.no_grad():
                loss = AllReduce.apply(reg_loss.sum()) / AllReduce.apply(
                    torch.tensor(sample_size, device=reg_loss.device)
                )
                if self.smooth_loss == 0:
                    self.smooth_loss = loss
                    self.smooth_loss2 = loss
                else:
                    d = self.cfg.adaptive_ema_decay_mom
                    self.smooth_loss = self.smooth_loss.float() * d + (1 - d) * loss
                    d2 = self.cfg.adaptive_ema_decay_mom2
                    self.smooth_loss2 = self.smooth_loss2.float() * d2 + (1 - d2) * loss
                    result["smooth_loss"] = self.smooth_loss
                    result["smooth_loss2"] = self.smooth_loss2

        return result

    def forward_decoder(
        self,
        x,
        decoder,
        encoder_mask,
        mask_indices,
        orig_B,
        orig_T,
        padding_mask,
    ):
        if self.cfg.mae_decoder_dropout > 0:
            x = F.dropout(x, self.cfg.mae_decoder_dropout, inplace=True)

        decoder_input = torch.zeros(
            (
                orig_B,
                orig_T,
                self.cfg.mae_decoder_dim
                if self.cfg.mae_encoder_decoder_attn
                and self.cfg.mae_decoder_layers == 0
                else self.cfg.encoder_embed_dim,
            ),
            dtype=x.dtype,
            device=x.device,
        )
        if not self.cfg.mae_decoder_zero_mask:
            if self.mask_emb is not None:
                masks = self.mask_emb
            else:
                masks = decoder_input.new_empty(
                    (*(decoder_input[mask_indices]).shape,)
                ).normal_(0, self.cfg.mask_noise_std)

            decoder_input = index_put(decoder_input, mask_indices, masks)

        MT = mask_indices[0].sum()
        B, T = (
            mask_indices.size(0),
            mask_indices.size(1) - MT,
        )

        dec_lrs = []
        alibi_bias_dec = None
        if self.cfg.use_alibi_decoder:
            assert self.cfg.mae_conv_decoder_layers <= 0
            alibi_bias_dec = get_alibi_bias(
                self.alibi_biases,
                self.cfg.mae_decoder_heads,
                batch_size=orig_B,
                time_steps=orig_T,
                dtype=x.dtype,
                device=x.device,
            )
        if self.cfg.mae_conv_decoder_layers > 0:
            decoder_input = index_put(
                decoder_input, ~mask_indices, x.reshape(-1, x.size(-1))
            ).transpose(1, 2)

            def add_residual(x, residual):
                if residual is None:
                    return x

                ret = x + residual

                if self.cfg.mae_conv_residual_avg:
                    ret = ret / 2
                if self.cfg.mae_conv_residual_ln:
                    ret = ret.transpose(1, 2)
                    ret = F.layer_norm(ret.float(), ret.shape[-1:]).type_as(ret)
                    ret = ret.transpose(1, 2)
                if self.cfg.mae_conv_residual_act:
                    ret = F.gelu(ret)

                return ret

            residual = None
            if self.decoder_in_proj is not None:
                if self.cfg.mae_use_in_proj_residual:
                    residual = decoder_input
                decoder_input = self.decoder_in_proj(decoder_input)
                decoder_input = add_residual(decoder_input, residual)

            residual = x = decoder_input
            if self.cfg.mae_conv_decoder_residual_context_only:
                resid_mask_indices = mask_indices.unsqueeze(1)

            for i, layer in enumerate(decoder):
                x = layer(x)
                if (
                    self.cfg.mae_conv_decoder_residual
                    and residual is not None
                    and residual.size(1) == x.size(1)
                ):
                    if self.cfg.mae_conv_decoder_residual_context_only:
                        residual = residual.masked_fill(resid_mask_indices, 0)
                    x = add_residual(x, residual)
                residual = x
                dec_lrs.append(x)
            x = x.transpose(1, 2)

            if self.post_decoder != None:
                x = self.post_decoder(x)
                if residual is not None:
                    if self.cfg.mae_conv_decoder_residual_context_only:
                        residual = residual.masked_fill(resid_mask_indices, 0)
                    x = add_residual(x, residual.transpose(1, 2))
            if self.post_decoder_ln is not None:
                x = self.post_decoder_ln(x)

        elif not self.cfg.mae_encoder_decoder_attn:
            assert not self.cfg.mae_no_self_attn

            decoder_input = index_put(
                decoder_input, ~mask_indices, x.reshape(-1, x.size(-1))
            )

            if self.cfg.mae_share_pos_enc:
                decoder_input = self.encoder.add_positions(decoder_input)

            if self.decoder_in_proj is not None:
                decoder_input = self.decoder_in_proj(decoder_input)
            x, _ = decoder(
                decoder_input, padding_mask=padding_mask, alibi_bias=alibi_bias_dec
            )

        else:
            if self.cfg.mae_decoder_pos_type == PosEmb.CONV:
                if self.decoder_in_proj is not None:
                    xp = self.decoder_in_proj(x)
                else:
                    xp = x

                decoder_input = index_put(
                    decoder_input,
                    ~encoder_mask,
                    xp.reshape(-1, xp.size(-1)),
                )

            decoder_input = self.decoder.add_positions(decoder_input)

            decoder_input = decoder_input[mask_indices].view(B, MT, -1)

            if alibi_bias_dec is not None:
                alibi_bias_dec = self.masked_alibi(
                    alibi_bias_dec, mask_indices, orig_B, orig_T
                )

            x, _ = decoder(
                decoder_input,
                padding_mask=padding_mask,
                skip_positions=True,
                encoder_out=x.transpose(0, 1),
                alibi_bias=alibi_bias_dec,
            )

        return x, dec_lrs

    def d2v_loss(self, x, y):
        if len(self.grad_mult_stack) > 0:
            m = self.grad_mult_stack.pop()
            GradMultiply.apply(x, m)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y, reduction="none")
        else:
            loss = F.smooth_l1_loss(
                x.float(), y, reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.cfg.use_l1:
            loss += F.l1_loss(x.float(), y, reduction="none")
            loss = loss / 2

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        if self.cfg.layer_to_layer:
            scale /= self.cfg.average_top_k_layers

        reg_loss = loss * scale

        sample_size = loss.size(0)

        return reg_loss, sample_size

    @torch.no_grad()
    def make_targets(self, y, num_layers, num_unmasked=0):
        if num_layers < 1:
            target_layer_results = [y["x"].transpose(0, 1)]
        else:
            if self.cfg.end_of_block_targets:
                target_layer_results = [l[1] for l in y["layer_results"]]
            else:
                target_layer_results = [l[2] for l in y["layer_results"]]

            if self.average_layer_list is None:
                target_layer_results = target_layer_results[-num_layers:]
            else:
                target_layer_results = [
                    target_layer_results[i]
                    for i in range(len(target_layer_results))
                    if i in self.average_layer_list
                ]

        if num_unmasked > 0:
            target_layer_results = [tl[:, :num_unmasked] for tl in target_layer_results]

        if self.cfg.target_layerdrop > 0:
            new_results = []
            probs = np.random.rand(len(target_layer_results))
            target_prob = self.cfg.target_layerdrop
            for p, tl in zip(probs, target_layer_results):
                if p >= target_prob:
                    new_results.append(tl)
            if len(new_results) == 0:
                new_results.append(random.choice(target_layer_results))
            target_layer_results = new_results

        permuted = False
        if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
            target_layer_results = [
                tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
            ]
            permuted = True
        if self.cfg.batch_norm_target_layer:
            target_layer_results = [
                F.batch_norm(
                    tl.float(), running_mean=None, running_var=None, training=True
                )
                for tl in target_layer_results
            ]
        if self.cfg.instance_norm_target_layer:
            target_layer_results = [
                F.instance_norm(tl.float()) for tl in target_layer_results
            ]
        if permuted:
            target_layer_results = [
                tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
            ]
        if self.cfg.group_norm_target_layer:
            target_layer_results = [
                F.layer_norm(tl.float(), tl.shape[-2:]) for tl in target_layer_results
            ]
        if self.cfg.layer_norm_target_layer:
            target_layer_results = [
                F.layer_norm(tl.float(), tl.shape[-1:]) for tl in target_layer_results
            ]

        y = target_layer_results[0].float()
        if self.cfg.layer_to_layer:
            y = torch.stack(target_layer_results, dim=-2)
        else:
            for tl in target_layer_results[1:]:
                y.add_(tl.float())
            y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.normalize_target_magnitude > 0:
            y = F.normalize(y.float(), dim=-1) * self.cfg.normalize_target_magnitude

        if self.cfg.clamp_target_magnitude > 0:
            y.clamp_(
                min=-self.cfg.clamp_target_magnitude,
                max=self.cfg.clamp_target_magnitude,
            )

        if self.cfg.tanh_targets_div > 0:
            y = (
                torch.tanh(y.div_(self.cfg.tanh_targets_div))
                * self.cfg.tanh_targets_mult
            )

        if self.cfg.instance_norm_targets:
            assert not self.cfg.layer_to_layer
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
        if not permuted:
            y = y.transpose(0, 1)
        return y

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

    def extract_features(self, source, padding_mask, mask=False, layer=None):
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        self.cfg.clone_batch = 1
        self.prototypes = None
        self.decoder = None
        self.order_head = None

        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
