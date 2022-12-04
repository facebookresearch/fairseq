# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum, auto

from omegaconf import II

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    Fp32InstanceNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    RelPositionalEncoding,
    SamePad,
    TransposeLast,
    MaskPowerNorm,
    PositionalEmbedding,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.conformer_layer import ConformerWav2Vec2EncoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor

from .utils import pad_to_multiple

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])
LAYER_TYPE_CHOICES = ChoiceEnum(["transformer", "conformer"])


class PosEmb(Enum):
    CONV = auto()
    LEARNED = auto()
    FIXED = auto()
    NONE = auto()
    CAPE = auto()
    FIXED_AND_CONV = auto()


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def sample_negatives(y, num, n_negatives, cross_sample_negatives, padding_count=None):

    if n_negatives == 0 and cross_sample_negatives == 0:
        return y.new(0)

    bsz, tsz, fsz = y.shape
    y = y.reshape(-1, fsz)  # BTC => (BxT)C

    # FIXME: what happens if padding_count is specified?
    cross_high = tsz * bsz
    high = tsz - (padding_count or 0)
    with torch.no_grad():
        assert high > 1, f"{bsz,tsz,fsz}"

        if n_negatives > 0:
            tszs = buffered_arange(num).unsqueeze(-1).expand(-1, n_negatives).flatten()

            neg_idxs = torch.randint(
                low=0, high=high - 1, size=(bsz, n_negatives * num)
            )
            neg_idxs[neg_idxs >= tszs] += 1

        if cross_sample_negatives > 0:
            tszs = (
                buffered_arange(num)
                .unsqueeze(-1)
                .expand(-1, cross_sample_negatives)
                .flatten()
            )

            cross_neg_idxs = torch.randint(
                low=0,
                high=cross_high - 1,
                size=(bsz, cross_sample_negatives * num),
            )
            cross_neg_idxs[cross_neg_idxs >= tszs] += 1

    if n_negatives > 0:
        neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
    else:
        neg_idxs = cross_neg_idxs

    if cross_sample_negatives > 0 and n_negatives > 0:
        neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

    negs = y[neg_idxs.view(-1)]
    negs = negs.view(bsz, num, n_negatives + cross_sample_negatives, fsz).permute(
        2, 0, 1, 3
    )  # to NxBxTxC
    return negs, neg_idxs


def compute_preds(x, y, negatives, logit_temp):

    neg_is_pos = (y == negatives).all(-1)

    y = y.unsqueeze(0)
    targets = torch.cat([y, negatives], dim=0)

    logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
    logits = logits / logit_temp

    if is_xla_tensor(logits) or neg_is_pos.any():
        if not hasattr(compute_preds, "_inftensor"):
            fillval = -float(2 ** 30)
            compute_preds._inftensor = (
                torch.tensor(fillval, device=x.device)
                if is_xla_tensor(logits)
                else float("-inf")
            )
        logits[1:] = index_put(logits[1:], neg_is_pos, compute_preds._inftensor)

    return logits


@dataclass
class Wav2Vec2Config(FairseqDataclass):
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )
    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )

    self_attn_norm_type: str = "layer"
    final_norm_type: str = "layer"
    force_deepnorm_init: bool = False

    mlp_encoder: bool = False
    mlp_layers: int = 3
    mlp_layernorm: bool = True

    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
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
    mask_min_space: int = field(
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
    add_masks: bool = False
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
    mask_channel_before: bool = False
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
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )
    pos_conv_ln: bool = False
    pos_emb_type: PosEmb = PosEmb.CONV
    residual_conv_pos: bool = False

    cape_content_scale: Optional[float] = 1.0
    cape_positions_delta: float = 0.02
    cape_max_global_shift: float = 20.0
    cape_max_local_shift: float = 0.5
    cape_max_global_scaling: float = 1.1
    cape_normalize: bool = True
    cape_freq_scale: float = 20.0

    num_pos: int = 3071

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})

    affine_encoder_norm: bool = True
    affine_extractor_norm: bool = True
    affine_norms: bool = True
    drop_path: float = 0
    debug: int = 0

    no_grads_after_layer: int = -1

    ddp_backend: str = II("distributed_training.ddp_backend")
    fsdp_mp: str = II("distributed_training.fsdp_mp")
    fp16: bool = II("common.fp16")


@register_model("wav2vec2", dataclass=Wav2Vec2Config)
class Wav2Vec2Model(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        if cfg.mlp_encoder:
            pass
        else:
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
                affine_norms=cfg.affine_norms,
            )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple

        self.mask_prob = cfg.mask_prob
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
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                    weight_proj_depth=cfg.quantizer_depth,
                    weight_proj_factor=cfg.quantizer_factor,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        encoder_cls = TransformerEncoder
        if cfg.layer_type == "conformer" and cfg.pos_enc_type in ["rel_pos", "rope"]:
            encoder_cls = ConformerEncoder

        self.encoder = encoder_cls(cfg)
        self.layer_norm = LayerNorm(self.embed, elementwise_affine=cfg.affine_norms)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

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

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                    add_masks=self.cfg.add_masks,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
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
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

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

        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            unmasked_features = unmasked_features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        if self.quantizer:
            if self.negatives_from_everywhere:
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)

                negs, _ = sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    n_negatives=self.n_negatives,
                    cross_sample_negatives=self.cross_sample_negatives,
                    padding_count=padding_count,
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.project_q(y)

                negs, _ = sample_negatives(
                    y,
                    y.size(1),
                    n_negatives=self.n_negatives,
                    cross_sample_negatives=self.cross_sample_negatives,
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = sample_negatives(
                    unmasked_features,
                    y.size(1),
                    n_negatives=self.n_negatives,
                    cross_sample_negatives=self.cross_sample_negatives,
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = sample_negatives(
                    y,
                    y.size(1),
                    n_negatives=self.n_negatives,
                    cross_sample_negatives=self.cross_sample_negatives,
                    padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = compute_preds(x, y, negs, self.logit_temp).type_as(x)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False, layer=None):
        res = self.forward(
            source, padding_mask, mask=mask, features_only=True, layer=layer
        )
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self, last_layer=None):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        affine_norms: bool = True,
        fsdp_fp16: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=affine_norms),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=affine_norms),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            bl = block(
                in_d,
                dim,
                k,
                stride,
                is_layer_norm=mode == "layer_norm",
                is_group_norm=mode == "default" and i == 0,
                conv_bias=conv_bias,
            )

            if fsdp_fp16:
                bl = bl.half().cuda()
            bl = fsdp_wrap(bl)

            self.conv_layers.append(bl)
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class MlpFeatureExtractionModel(nn.Module):
    def __init__(self, n_in, n_out, n_layers, layer_norm):
        super().__init__()

        layers = []

        self.n_in = n_in

        curr_d = n_in
        for i in range(n_layers - 1):
            layers.append(nn.Linear(curr_d, n_out))
            if layer_norm:
                layers.append(Fp32LayerNorm(n_out))
            elif i == 0:
                layers.append(
                    nn.Sequential(
                        TransposeLast(),
                        Fp32GroupNorm(n_out, n_out),
                        TransposeLast(),
                    ),
                )
            layers.append(nn.GELU())
            curr_d = n_out
        layers.append(nn.Linear(curr_d, n_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        B, T = x.shape

        mod = T % self.n_in
        if mod > 0:
            x = F.pad(x, (0, self.n_in - mod))

        x = x.view(B, -1, self.n_in)

        x = self.net(x)

        return x


def make_conv_pos(e, k, g):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv


class TransformerEncoder(nn.Module):
    def build_encoder_layer(
        self,
        args: Wav2Vec2Config,
        encoder_dim=None,
        self_attn=True,
        ffn=True,
        norm1=True,
        norm2=True,
        enc_dec_resid=True,
    ):
        if args.layer_type == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
                self_attn_norm_type=args.self_attn_norm_type,
                final_norm_type=args.final_norm_type,
                affine_norms=args.affine_norms,
                debug=args.debug,
                encoder_dim=encoder_dim,
                self_attn=self_attn,
                ffn=ffn,
                norm1=norm1,
                norm2=norm2,
                enc_dec_resid=enc_dec_resid,
                drop_path=args.drop_path,
            )
        elif args.layer_type == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                activation_fn="swish",
                attn_type=args.attn_type,
                use_fp16=args.fp16,
                pos_enc_type="abs",
            )
        if args.ddp_backend == "fully_sharded" and args.fp16:
            layer = layer.half().cuda()
        layer = fsdp_wrap(layer)
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(
        self,
        args: Wav2Vec2Config,
        encoder_dim=None,
        self_attn=True,
        ffn=True,
        norm1=True,
        norm2=True,
        pre_norm=True,
        enc_dec_resid=True,
    ):
        super().__init__()

        self.args = args
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple
        self.pos_emb = None
        self.pos_conv = None
        self.residual_conv_pos = args.residual_conv_pos

        self.has_encoder_states = encoder_dim is not None and encoder_dim > 0

        if args.pos_emb_type == PosEmb.NONE:
            pass
        elif (
            args.pos_emb_type == PosEmb.CONV
            or args.pos_emb_type == PosEmb.FIXED_AND_CONV
        ):
            pos_conv_depth = getattr(args, "pos_conv_depth", 1)
            if pos_conv_depth > 1:
                num_layers = args.pos_conv_depth
                k = max(3, args.conv_pos // num_layers)

                def make_conv_block(e, k, g, l):
                    return nn.Sequential(
                        *[
                            nn.Sequential(
                                nn.Conv1d(
                                    e,
                                    e,
                                    kernel_size=k,
                                    padding=k // 2,
                                    groups=g,
                                ),
                                SamePad(k),
                                TransposeLast(),
                                LayerNorm(e, elementwise_affine=False),
                                TransposeLast(),
                                nn.GELU(),
                            )
                            for _ in range(l)
                        ]
                    )

                self.pos_conv = make_conv_block(
                    self.embedding_dim, k, args.conv_pos_groups, num_layers
                )
                if args.pos_conv_ln:
                    self.pos_conv = nn.Sequential(
                        LayerNorm(self.embedding_dim), self.pos_conv
                    )

            else:
                self.pos_conv = make_conv_pos(
                    self.embedding_dim,
                    args.conv_pos,
                    args.conv_pos_groups,
                )

            if args.pos_emb_type == PosEmb.FIXED_AND_CONV:
                self.pos_emb = PositionalEmbedding(
                    num_embeddings=args.num_pos + 1,
                    embedding_dim=self.embedding_dim,
                    padding_idx=0,
                    learned=False,
                )
        elif args.pos_emb_type == PosEmb.CAPE:
            from cape import CAPE1d

            self.pos_emb = CAPE1d(
                d_model=args.encoder_embed_dim,
                max_global_shift=args.cape_max_global_shift,
                max_local_shift=args.cape_max_local_shift,
                max_global_scaling=args.cape_max_global_scaling,
                normalize=args.cape_normalize,
                freq_scale=args.cape_freq_scale,
                batch_first=True,
            )
            if args.cape_content_scale is not None:
                self.pos_emb.set_content_scale(args.cape_content_scale)

        else:
            self.pos_emb = PositionalEmbedding(
                num_embeddings=args.num_pos + 1,
                embedding_dim=self.embedding_dim,
                padding_idx=0,
                learned=(args.pos_emb_type == PosEmb.LEARNED),
            )

        self.layers = nn.ModuleList(
            [
                self.build_encoder_layer(
                    args, encoder_dim, self_attn, ffn, norm1, norm2, enc_dec_resid
                )
                for _ in range(args.encoder_layers)
            ]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = (
            LayerNorm(
                self.embedding_dim,
                elementwise_affine=args.affine_encoder_norm and args.affine_norms,
            )
            if pre_norm
            else None
        )

        self.layerdrop = args.encoder_layerdrop
        self.no_grads_after_layer = args.no_grads_after_layer

        self.apply(init_bert_params)

        for layer in self.layers:
            if hasattr(layer, "custom_init"):
                layer.custom_init(args.force_deepnorm_init)

    def forward(
        self,
        x,
        padding_mask=None,
        layer=None,
        skip_positions=False,
        encoder_out=None,
        alibi_bias=None,
    ):

        x, layer_results = self.extract_features(
            x,
            padding_mask,
            layer,
            skip_positions=skip_positions,
            encoder_out=encoder_out,
            alibi_bias=alibi_bias,
        )

        if self.layer_norm_first and layer is None and self.layer_norm is not None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
        skip_positions=False,
        encoder_out=None,
        alibi_bias=None,
    ):

        assert encoder_out is None or self.has_encoder_states
        # TODO encoder padding

        if padding_mask is not None:
            if x.size(1) > padding_mask.size(1):
                padding_mask = F.pad(padding_mask, (1, 0), value=False)
            x = index_put(x, padding_mask, 0)

        if not skip_positions:
            x = self.add_positions(x)

        if not self.layer_norm_first and self.layer_norm is not None:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )

        if pad_length > 0 and alibi_bias is not None:
            alibi_bias = F.pad(alibi_bias, (0, pad_length, 0, pad_length))

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    encoder_out=encoder_out,
                    alibi_bias=alibi_bias,
                )

                if i >= min_layer:
                    layer_results.append((z, x, lr))

            if self.no_grads_after_layer == i:
                x = GradMultiply.apply(x, 0)

            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:, :-pad_length, :-pad_length] if a is not None else a,
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length] if c is not None else c,
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def add_positions(self, x):
        if self.pos_emb is not None:
            if self.args.pos_emb_type == PosEmb.CAPE:
                x = self.pos_emb(
                    x, positions_delta=self.args.cape_positions_delta
                ).type_as(x)
            else:
                pos_inds = utils.buffered_arange(x.size(1) + 1)[1:]
                pos_inds = (
                    pos_inds.unsqueeze(0).expand(x.size(0), -1).to(device=x.device)
                )
                pos_emb = self.pos_emb(pos_inds)
                x = x + pos_emb

        if self.pos_conv is not None:
            if self.residual_conv_pos:
                residual = px = x.transpose(1, 2)
                for block in self.pos_conv:
                    px = block(px)
                    px = px + residual
                    residual = px
                x = x + px.transpose(1, 2)
            else:
                x_conv = self.pos_conv(x.transpose(1, 2))
                x_conv = x_conv.transpose(1, 2)
                x = x + x_conv
        return x

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class ConformerEncoder(TransformerEncoder):
    def build_encoder_layer(self, args):
        layer = ConformerWav2Vec2EncoderLayer(
            embed_dim=self.embedding_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
            activation_fn="swish",
            attn_type=args.attn_type,
            pos_enc_type=args.pos_enc_type,
            use_fp16=args.fp16,  # only used for rope
        )
        layer = fsdp_wrap(layer)
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.pos_enc_type = args.pos_enc_type
        max_source_positions = self.max_positions()

        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                max_source_positions, self.embedding_dim
            )
        elif self.pos_enc_type == "rope":
            self.embed_positions = None
        else:
            raise Exception("Unsupported positional encoding type")

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(
            self.embedding_dim, elementwise_affine=args.affine_norms
        )
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # B X T X C here
        position_emb = None
        if self.pos_enc_type == "rel_pos":
            position_emb = self.embed_positions(x)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    position_emb=position_emb,
                )
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        self_attn_norm_type: str = "layer",
        final_norm_type: str = "layer",
        affine_norms: bool = True,
        debug: int = 0,
        encoder_dim: Optional[int] = None,
        self_attn: bool = True,
        ffn=True,
        norm1=True,
        norm2=True,
        enc_dec_resid=True,
        drop_path=0.0,
        norm_eps=1e-5,
        single_qkv=False,
        fused_ln=True,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.drop_path = drop_path
        self.activation_dropout = activation_dropout
        self.enc_dec_resid = enc_dec_resid

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)

        if self_attn:
            self.self_attn = MultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
                qkv_single_proj=single_qkv,
                debug=debug,
            )
        else:
            self.self_attn = None

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.alpha = 1
        self.alpha2 = 1
        if self_attn:
            if self_attn_norm_type == "instance":
                self.self_attn_layer_norm = nn.Sequential(
                    TransposeLast(),
                    Fp32InstanceNorm(self.embedding_dim),
                    TransposeLast(),
                )
            elif self_attn_norm_type == "power":
                self.self_attn_layer_norm = MaskPowerNorm(self.embedding_dim)
            elif self_attn_norm_type == "layer":
                # layer norm associated with the self attention layer
                self.self_attn_layer_norm = LayerNorm(
                    self.embedding_dim,
                    elementwise_affine=affine_norms,
                    eps=norm_eps,
                    export=not fused_ln,
                )
            elif self_attn_norm_type == "deepnorm":
                self.self_attn_layer_norm = LayerNorm(
                    self.embedding_dim,
                    elementwise_affine=affine_norms,
                    eps=norm_eps,
                    export=not fused_ln,
                )
                if embedding_dim == 768:
                    self.alpha = 24 ** 0.25
                elif embedding_dim == 1024:
                    self.alpha = 48 ** 0.25
                else:
                    raise Exception()
            else:
                raise Exception(f"unknown norm type {self_attn_norm_type}")

        self.encoder_attn_layer_norm = None
        self.encoder_attn = None
        if encoder_dim is not None and encoder_dim > 0:
            self.encoder_attn = MultiheadAttention(
                embedding_dim,
                num_attention_heads,
                kdim=encoder_dim,
                vdim=encoder_dim,
                dropout=attention_dropout,
                encoder_decoder_attention=True,
            )
            if self_attn_norm_type == "deepnorm":
                self.encoder_attn_layer_norm = (
                    LayerNorm(
                        self.embedding_dim,
                        elementwise_affine=affine_norms,
                        eps=norm_eps,
                        export=not fused_ln,
                    )
                    if norm1
                    else None
                )
                if embedding_dim == 768:
                    self.alpha = 24 ** 0.25
                elif embedding_dim == 1024:
                    self.alpha = 48 ** 0.25
                else:
                    raise Exception()
            else:
                self.encoder_attn_layer_norm = (
                    LayerNorm(
                        embedding_dim,
                        elementwise_affine=affine_norms,
                        eps=norm_eps,
                        export=not fused_ln,
                    )
                    if norm1
                    else None
                )
        else:
            self.encoder_attn = None

        if ffn:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
            self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        else:
            self.fc1 = None
            self.fc2 = None

        if not norm2:
            self.final_layer_norm = None
        elif final_norm_type == "instance":
            self.final_layer_norm = nn.Sequential(
                TransposeLast(), Fp32InstanceNorm(self.embedding_dim), TransposeLast()
            )
        elif final_norm_type == "power":
            self.final_layer_norm = MaskPowerNorm(self.embedding_dim)
        elif final_norm_type == "layer":
            # layer norm associated with the self attention layer
            self.final_layer_norm = LayerNorm(
                self.embedding_dim,
                elementwise_affine=affine_norms,
                eps=norm_eps,
                export=not fused_ln,
            )
        elif final_norm_type == "deepnorm":
            self.final_layer_norm = LayerNorm(
                self.embedding_dim,
                elementwise_affine=affine_norms,
                eps=norm_eps,
                export=not fused_ln,
            )
            if embedding_dim == 768:
                self.alpha2 = 24 ** 0.25
            elif embedding_dim == 1024:
                self.alpha2 = 48 ** 0.25
            else:
                raise Exception()
        else:
            raise Exception(f"unknown norm type {self_attn_norm_type}")

    def custom_init(self, force_dn=False):
        if self.alpha != 1 or self.alpha2 != 1 or force_dn:
            if self.embedding_dim == 768:
                beta = (8 * 12) ** -0.25
            elif self.embedding_dim == 1024:
                beta = (8 * 24) ** -0.25
            else:
                raise Exception()

            nn.init.xavier_normal_(self.fc1.weight, gain=beta)
            nn.init.xavier_normal_(self.fc2.weight, gain=beta)

            if self.self_attn is not None:
                nn.init.xavier_normal_(self.self_attn.v_proj.weight, gain=beta)
                nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=beta)
                nn.init.xavier_normal_(self.self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(self.self_attn.k_proj.weight, gain=1)

            if self.encoder_attn_layer_norm is not None:
                nn.init.xavier_normal_(self.encoder_attn.v_proj.weight, gain=beta)
                nn.init.xavier_normal_(self.encoder_attn.out_proj.weight, gain=beta)
                nn.init.xavier_normal_(self.encoder_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(self.encoder_attn.k_proj.weight, gain=1)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        encoder_out=None,
        encoder_padding_mask=None,
        alibi_bias=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            if self.self_attn is not None:
                x = self.self_attn_layer_norm(x)
                x, attn = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=self_attn_padding_mask,
                    attn_mask=self_attn_mask,
                    need_weights=need_weights,
                    alibi_bias=alibi_bias,
                )
                x = self.dropout1(x)
                x = residual * self.alpha + drop_path(x, self.drop_path, self.training)

            if self.encoder_attn is not None:
                assert encoder_out is not None
                residual = x
                if self.encoder_attn_layer_norm is not None:
                    x = self.encoder_attn_layer_norm(x)
                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    static_kv=True,
                    need_weights=False,
                    need_head_weights=False,
                )
                x = self.dropout1(x)
                if self.enc_dec_resid:
                    x = residual * self.alpha + drop_path(
                        x, self.drop_path, self.training
                    )

            residual = x
            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)
            if self.fc1 is not None:
                x = self.activation_fn(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)

            if self.fc1 is not None:
                x = residual * self.alpha2 + drop_path(x, self.drop_path, self.training)
        else:
            if self.self_attn is not None:
                x, attn = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=self_attn_padding_mask,
                    need_weights=need_weights,
                    alibi_bias=alibi_bias,
                )

                x = self.dropout1(x)
                x = residual * self.alpha + drop_path(x, self.drop_path, self.training)

                x = self.self_attn_layer_norm(x)

            if self.encoder_attn is not None:
                assert encoder_out is not None
                residual = x
                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    static_kv=True,
                    need_weights=False,
                    need_head_weights=False,
                )
                x = self.dropout1(x)
                if self.enc_dec_resid:
                    x = residual * self.alpha + drop_path(
                        x, self.drop_path, self.training
                    )

                if self.encoder_attn_layer_norm is not None:
                    x = self.encoder_attn_layer_norm(x)

            residual = x
            if self.fc1 is not None:
                x = self.activation_fn(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            if self.fc1 is not None:
                x = residual * self.alpha2 + drop_path(x, self.drop_path, self.training)
            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)

        return x, (attn, layer_result)
