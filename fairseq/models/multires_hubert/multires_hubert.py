# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import math
import torch.nn as nn
from omegaconf import II
from fairseq.models.wav2vec.wav2vec import norm_block

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from omegaconf import II, MISSING, open_dict
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.multires_hubert_pretraining import (
    MultiresHubertPretrainingConfig,
    MultiresHubertPretrainingTask,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiresHubertConfig(FairseqDataclass):
    label_rate: float = II("task.label_rate")
    #     label_rate: 1,2,2,5
    #                 (imply (1,2), (2,5))
    #     if base label_rate = 50
    #     (1,2), (2,5) --> label rates 50, 25, 10
    label_rate_ratios: List[int] = field(
        default=MISSING, metadata={"help": "tuple for label rates e.g., [(1,2), (2,5)]"}
    )

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    # the blocks for each label rate
    encoder_layers: int = field(
        default="2",
        metadata={
            "help": "num encoder layers in the each block (one sub module of the U-net)"
        },
    )
    override_encoder_layers: str = field(
        default="",
        metadata={
            "help": "specific layer numbers for each block (one sub module of the U-net) for the training"
        },
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
    conv_adapator_kernal: int = field(
        default=7, metadata={"help": "kernal size for conv adaptor"}
    )
    use_plain_updownsample: bool = field(
        default=False, metadata={"help": "whether to use plain up downsample"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
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
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=True,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )
    use_single_target: bool = field(
        default=False,
        metadata={
            "help": "whether to use single data (in that case, we will compute with the fixed label rate)"
        },
    )
    use_single_prediction: bool = field(
        default=False,
        metadata={
            "help": "if true, we will not conduct mlm prediction in low resolution in the middle"
        },
    )
    use_multi_stream: bool = field(
        default=False,
        metadata={
            "help": "whether to use multi-stream setting (in this setting, we have multiple streams with the same resolution)"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
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

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

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


@register_model("multires_hubert", dataclass=MultiresHubertConfig)
class MultiresHubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: MultiresHubertConfig,
        task_cfg: MultiresHubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"MultiresHubertModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        # Estimate label rates
        assert (
            cfg.label_rate_ratios != "None"
        ), "without ratios, the model is exactly as the Hubert model"
        self.label_rate_ratios = []
        self.base_rate = cfg.label_rate
        self.label_rates = []
        self.downsample_modules = nn.ModuleList()
        self.upsample_modules = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.use_single_target = cfg.use_single_target
        self.use_single_prediction = cfg.use_single_prediction
        self.use_plain_updownsample = cfg.use_plain_updownsample

        # For decide the override encoder layers, so that the layer number is not equally distributed
        if cfg.override_encoder_layers != "":
            self.override_encoder_layers = eval(cfg.override_encoder_layers)
            assert (
                len(self.override_encoder_layers) % 2 == 1
            ), "must be odd number of layers if specify detailed layers"
            assert (
                len(self.override_encoder_layers) // 2
                == len(cfg.label_rate_ratios) // 2
            ), "number of override encoder layers must match the label rate ratios information"
            self.len_encoder_modules = len(self.override_encoder_layers)
        else:
            self.override_encoder_layers = None
            self.len_encoder_modules = None

        # use different layers instead of equally distributed ones
        middle_override_encoder_layer = (
            self.override_encoder_layers[self.len_encoder_modules // 2]
            if self.override_encoder_layers is not None
            else None
        )
        skip_middle_pos_conv = False if len(cfg.label_rate_ratios) < 2 else True

        self.middle_encoder = TransformerEncoder(
            cfg,
            skip_pos_conv=skip_middle_pos_conv,
            override_encoder_layer=middle_override_encoder_layer,
        )

        first_pos_conv = False  # only enable pos_conv for the first encoder
        raw_label_rate_ratios = cfg.label_rate_ratios
        for i in range(len(raw_label_rate_ratios) // 2):
            # check if have override encoder layers
            if self.override_encoder_layers is not None:
                override_encoder_layer = self.override_encoder_layers[i]
                override_decoder_layer = self.override_encoder_layers[
                    self.len_encoder_modules - 1 - i
                ]
            else:
                override_encoder_layer, override_decoder_layer = None, None

            self.label_rate_ratios.append(
                (raw_label_rate_ratios[i * 2], raw_label_rate_ratios[i * 2 + 1])
            )
            if self.use_plain_updownsample:
                self.downsample_modules.append(
                    ConvDownsampler(
                        k=cfg.conv_adapator_kernal,
                        label_rate=(
                            (
                                raw_label_rate_ratios[i * 2],
                                raw_label_rate_ratios[i * 2 + 1],
                            )
                        ),
                        dropout=0.0,
                        channels=cfg.encoder_embed_dim,
                        activation=nn.GELU(),
                        log_compression=False,
                        skip_connections=True,
                        highway=True,
                        residual_scale=0.4,
                    )
                )
            else:
                self.downsample_modules.append(
                    ConvAdapter(
                        k=cfg.conv_adapator_kernal,
                        label_rate=(
                            (
                                raw_label_rate_ratios[i * 2],
                                raw_label_rate_ratios[i * 2 + 1],
                            )
                        ),
                        dropout=0.0,
                        channels=cfg.encoder_embed_dim,
                        activation=nn.GELU(),
                        log_compression=False,
                        skip_connections=True,
                        highway=True,
                        residual_scale=0.4,
                    )
                )
            if not first_pos_conv:
                self.encoders.append(
                    TransformerEncoder(
                        cfg, override_encoder_layer=override_encoder_layer
                    )
                )  # TODO(jiatong): add conformer options
                first_pos_conv = True
            else:
                self.encoders.append(
                    TransformerEncoder(
                        cfg,
                        skip_pos_conv=True,
                        override_encoder_layer=override_encoder_layer,
                    )
                )
            if self.use_plain_updownsample:
                self.upsample_modules.append(
                    ConvUpsampler(
                        k=cfg.conv_adapator_kernal,
                        label_rate=(
                            (
                                raw_label_rate_ratios[i * 2 + 1],
                                raw_label_rate_ratios[i * 2],
                            )
                        ),
                        dropout=0.0,
                        channels=cfg.encoder_embed_dim,
                        activation=nn.GELU(),
                        log_compression=False,
                        skip_connections=True,
                        highway=True,
                        residual_scale=0.4,
                    )
                )
            else:
                self.upsample_modules.append(
                    ConvAdapter(
                        k=cfg.conv_adapator_kernal,
                        label_rate=(
                            (
                                raw_label_rate_ratios[i * 2 + 1],
                                raw_label_rate_ratios[i * 2],
                            )
                        ),
                        dropout=0.0,
                        channels=cfg.encoder_embed_dim,
                        activation=nn.GELU(),
                        log_compression=False,
                        skip_connections=True,
                        highway=True,
                        residual_scale=0.4,
                    )
                )
            self.decoders.append(
                TransformerEncoder(
                    cfg,
                    skip_pos_conv=True,
                    override_encoder_layer=override_decoder_layer,
                )
            )

        base_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feature_ds_rates = [base_ds_rate]
        running_rate = self.base_rate

        if cfg.use_single_target or cfg.use_multi_stream:
            self.label_rates = self.base_rate
        else:
            self.label_rates.append(self.base_rate)

        for label_rate_ratio in self.label_rate_ratios:
            upsample_rate, downsample_rate = label_rate_ratio
            if (base_ds_rate * upsample_rate) % downsample_rate != 0:
                logger.warning(
                    "base rate: {} cannot be ideally processed with downsample rate {}".format(
                        base_ds_rate, downsample_rate
                    )
                )

            base_ds_rate = base_ds_rate * downsample_rate // upsample_rate
            self.feature_ds_rates.append(base_ds_rate)

            if not cfg.use_single_target and not cfg.use_multi_stream:
                running_rate = running_rate * upsample_rate // downsample_rate
                self.label_rates.append(running_rate)
        self.label_nums = len(
            self.feature_ds_rates
        )  # the number of labels for prediction (activate at iter 2)

        if type(self.label_rates) == float:
            self.feat2tar_ratios = [
                self.feature_ds_rates[i] * self.label_rates / task_cfg.sample_rate
                for i in range(len(self.feature_ds_rates))
            ]
        else:
            self.feat2tar_ratios = [
                self.feature_ds_rates[i] * self.label_rates[i] / task_cfg.sample_rate
                for i in range(len(self.feature_ds_rates))
            ]

        # self.feat2tar_ratios = self.feat2tar_ratios[::-1]

        # An running example of the label rate:
        #     base_ds_rate = 320
        #     self.label_rate_ratios = [(1, 2)]
        #     self.feature_ds_rates = [320, 640]
        #     self.label_rates = [50, 25]
        #     self.feat2tar_ratios = [1, 1]

        # Another running example of the label rate:
        #     base_ds_rate = 320
        #     self.label_rate_ratios = [(1, 2)]
        #     self.feature_ds_rates = [320, 640]
        #     self.label_rates = 100
        #     self.feat2tar_ratios = [4, 2]
        #     self.use_sinlge_target = True

        logging.info(
            "ds_rates: {}, label_rates: {}, feat2tar_ratios: {}".format(
                self.feature_ds_rates, self.label_rates, self.feat2tar_ratios
            )
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        # Note(jiatong): different from hubert, we just set the final dim as encoder_embed_dim
        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.layer_norm = LayerNorm(self.embed)

        self.predictor_head_num = 1 if self.use_single_prediction else self.label_nums

        self.target_glu = None
        if cfg.target_glu:
            self.target_glus = nn.ModuleList()
            for i in range(self.predictor_head_num):
                self.target_glus.append(
                    nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())
                )

        self.untie_final_proj = cfg.untie_final_proj
        self.final_projs = nn.ModuleList()

        # Note(jiatong): we do not have untie cases for multires hubert
        for i in range(self.predictor_head_num):
            self.final_projs.append(nn.Linear(cfg.encoder_embed_dim, final_dim))

        # modules below are not needed during fine-tuning
        self.multires_classes = []
        self.label_embs_concat = nn.ParameterList()

        for i in range(self.predictor_head_num):
            if self.use_single_target:
                num_classes = len(dictionaries[0])
            else:
                num_classes = len(dictionaries[i])
            self.multires_classes.append(num_classes)
            self.label_embs_concat.append(
                nn.Parameter(torch.FloatTensor(num_classes, final_dim))
            )
            nn.init.uniform_(self.label_embs_concat[i])

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(
        cls, cfg: MultiresHubertConfig, task: MultiresHubertPretrainingTask
    ):
        """Build a new model instance."""

        model = MultiresHubertModel(cfg, task.cfg, task.dictionaries)
        return model

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
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
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
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

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        feat2tar_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels

        feat_tsz = features.size(1)

        # skip if no target is provided
        if target is None:
            return features, None, None
        targ_tsz = target.size(1)
        if feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / feat2tar_ratio)
            features = features[:, :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * feat2tar_ratio
        target = target[:, target_inds.long()]
        return features, target

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        def align_size_sum(feat1, pad1, feat2):
            assert (
                abs(feat1.size(1) - feat2.size(1)) < 10
            ), "misaligned results for feat1 and feat2 of size {} - {}".format(
                feat1.size(1), feat2.size(1)
            )
            common_size = min(feat1.size(1), feat2.size(1))

            return (
                feat1[:, :common_size] + feat2[:, :common_size],
                pad1[:, :common_size],
            )

        # process encoders
        res_outputs = []  # final output for different resolution
        multi_mask_indices = []  # mask indices for different resolution
        residuals = []  # record the x in encoders
        padding_masks = []  # final padding masks
        # The encoder has (self.label_nums - 1) blocks
        for i in range(self.label_nums - 1):
            x, _ = self.encoders[i](x, padding_mask=padding_mask, layer=None)
            residuals.append(x)
            x, padding_mask, mask_indices = self.downsample_modules[i](
                x, padding=padding_mask, mask_indices=mask_indices
            )

        residual = self.middle_encoder(x, padding_mask=padding_mask, layer=None)[0]
        x = x + residual
        res_outputs.append(x)

        # process decoders
        # The encoder has (self.label_nums - 1) blocks
        padding_masks.append(padding_mask)
        multi_mask_indices.append(mask_indices)
        residuals.reverse()  # NOTE(jiatong): reverse res_output to match corresponding input
        for i in range(self.label_nums - 1):
            x, padding_mask, mask_indices = self.upsample_modules[
                self.label_nums - 2 - i
            ](x, padding=padding_mask, mask_indices=mask_indices)
            x, _ = self.decoders[i](x, padding_mask=padding_mask, layer=None)
            x, padding_mask = align_size_sum(x, padding_mask, residuals[i])
            res_outputs.append(x)
            padding_masks.append(padding_mask)
            multi_mask_indices.append(mask_indices)

        # NOTE(jiatong): need reverse of target list to allow matched target-representation
        res_outputs.reverse()
        padding_masks.reverse()
        multi_mask_indices.reverse()
        if target_list is not None:
            new_target_list = []
            for i in range(self.label_nums):
                if self.use_single_target:
                    res_outputs[i], reformat_target_list = self.forward_targets(
                        res_outputs[i], target_list[0], self.feat2tar_ratios[i]
                    )
                    new_target_list.append(reformat_target_list)
                else:
                    if target_list[i] is not None:
                        res_outputs[i], reformat_target_list = self.forward_targets(
                            res_outputs[i], target_list[i], self.feat2tar_ratios[i]
                        )
                        new_target_list.append(reformat_target_list)
                    else:
                        # Append a None target list then it won't be used to calculate loss
                        new_target_list.append(None)
                if padding_masks[i] is not None:
                    padding_masks[i] = self.forward_padding_mask(
                        res_outputs[i], padding_masks[i]
                    )
                if multi_mask_indices[i] is not None:
                    multi_mask_indices[i] = self.forward_padding_mask(
                        res_outputs[i], multi_mask_indices[i]
                    )


        if features_only:
            # NOTE(jiatong): need to reverse back
            res_outputs.reverse()
            return {
                "x": res_outputs,
                "padding_mask": padding_masks[0],
                "features": features,
            }

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        logit_m_list, logit_u_list = [], []
        for j in range(self.label_nums):
            if new_target_list[j] is None:
                continue  # skip empty targets
            label_embs_list = self.label_embs_concat[j].split(
                [self.multires_classes[j]], 0
            )
            # set the variables (after the set, the procedure is the same as hubert)
            # all the elements are list with only one element (to simulate the normal hubert process)
            x = res_outputs[j]
            target = new_target_list[j]
            padding_mask = padding_masks[j]
            mask_indices = multi_mask_indices[j]
            final_proj = self.final_projs[j]

            if not self.skip_masked:
                masked_indices = torch.logical_and(~padding_mask, mask_indices)
                proj_x_m = final_proj(x[masked_indices])
                logit_m_list.append(
                    compute_pred(proj_x_m, target[masked_indices], label_embs_list[0])
                )
            else:
                logit_m_list.append(None)

            if not self.skip_nomask:
                nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
                proj_x_u = final_proj(x[nomask_indices])
                logit_u_list.append(
                    compute_pred(proj_x_u, target[nomask_indices], label_embs_list[0])
                )
            else:
                logit_u_list.append(None)

            # if we only want one prediction, we can exit now
            if self.predictor_head_num == 1:
                break

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        last_layer: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        if last_layer:
            feature = feature[-1]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None


class ConvAdapter(nn.Module):
    """Conv adapter that combines two modules with different label rate with downsample or upsample.
    To allow different ratios than integer, two convs are utilized with first to upsample (numerator)
    and the second to downsample (denominator)"""

    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        log_compression=False,
        skip_connections=True,
        highway=True,
        residual_scale=0.4,
        non_affine_group_norm=False,
    ):
        super().__init__()

        def downsample_block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.Conv1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=(k - 1) // 2,
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        def upsample_block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.ConvTranspose1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=0,  # padding=(k - 1) // 2,
                    output_padding=(stride - 1),
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        # Lout =(Lin~H~R1)~Wstride~H~R2~Wpadding+dilation~W(kernel_size~H~R1)+output_padding+1
        self.upsample_conv = upsample_block(channels, k, label_rate[0])
        self.downsample_conv = downsample_block(channels, k, label_rate[1])

        self.upsample_rate, self.downsample_rate = label_rate
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.highway = highway
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x, padding=None, mask_indices=None):
        # Assume x1 = (B, T, C) as input
        x = x.permute(0, 2, 1)
        residual_before_upsample = x
        x = self.upsample_conv(x)
        upsample_size = x.size(2)

        # conduct upsample
        if self.skip_connections:
            residual_upsample = torch.repeat_interleave(
                residual_before_upsample, self.upsample_rate, dim=2
            )
            upsample_size = min(upsample_size, residual_upsample.size(2))
            x = (
                x[..., :upsample_size] + residual_upsample[..., :upsample_size]
            ) * self.residual_scale

        residual_before_downsample = x
        x = self.downsample_conv(x)
        downsample_size = x.size(2)

        if self.skip_connections:
            residual_downsample = residual_before_downsample[
                ..., :: self.downsample_rate
            ]
            downsample_size = min(x.size(2), residual_downsample.size(2))
            x = (
                x[..., :downsample_size] + residual_downsample[..., :downsample_size]
            ) * self.residual_scale

        if self.highway:
            residual_after_sample = residual_upsample[..., :: self.downsample_rate]
            final_size = min(x.size(2), residual_after_sample.size(2))
            x = (
                x[..., :final_size] + residual_after_sample[..., :final_size]
            ) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        x = x.permute(0, 2, 1)

        # process padding
        if padding is not None:
            padding = torch.repeat_interleave(padding, self.upsample_rate, dim=1)
            padding = padding[..., :: self.downsample_rate]
            padding = padding[..., : x.size(1)]

        # process mask indices
        if mask_indices is not None:
            mask_indices = torch.repeat_interleave(
                mask_indices, self.upsample_rate, dim=1
            )
            mask_indices = mask_indices[..., :: self.downsample_rate]
            mask_indices = mask_indices[..., : x.size(1)]
        return x, padding, mask_indices


class ConvDownsampler(nn.Module):
    """Conv downsampler that combines two modules with different label rate with downsample or upsample.
    To allow different ratios than integer, two convs are utilized with first to upsample (numerator)
    and the second to downsample (denominator)"""

    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        log_compression=False,
        skip_connections=True,
        highway=True,
        residual_scale=0.4,
        non_affine_group_norm=False,
    ):
        super().__init__()

        def downsample_block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.Conv1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=(k - 1) // 2,
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        self.downsample_conv = downsample_block(channels, k, label_rate[1])

        upsample_rate, self.downsample_rate = label_rate
        assert upsample_rate == 1, "must be 1 to perform downsample only"
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.highway = highway  # Useless as placeholder
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x, padding=None, mask_indices=None):
        # Assume x1 = (B, T, C) as input
        x = x.permute(0, 2, 1)

        residual_before_downsample = x
        x = self.downsample_conv(x)
        downsample_size = x.size(2)

        if self.skip_connections:
            residual_downsample = residual_before_downsample[
                ..., :: self.downsample_rate
            ]
            downsample_size = min(x.size(2), residual_downsample.size(2))
            x = (
                x[..., :downsample_size] + residual_downsample[..., :downsample_size]
            ) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        x = x.permute(0, 2, 1)

        # process padding
        if padding is not None:
            padding = padding[..., :: self.downsample_rate]
            padding = padding[..., : x.size(1)]

        # process mask indices
        if mask_indices is not None:
            mask_indices = mask_indices[..., :: self.downsample_rate]
            mask_indices = mask_indices[..., : x.size(1)]
        return x, padding, mask_indices


class ConvUpsampler(nn.Module):
    """Conv upsampler that combines two modules with different label rate with downsample or upsample.
    To allow different ratios than integer, two convs are utilized with first to upsample (numerator)
    and the second to downsample (denominator)"""

    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        log_compression=False,
        skip_connections=True,
        highway=True,
        residual_scale=0.4,
        non_affine_group_norm=False,
    ):
        super().__init__()

        def upsample_block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.ConvTranspose1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=0,  # padding=(k - 1) // 2,
                    output_padding=(stride - 1),
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        # Lout =(Lin~H~R1)~Wstride~H~R2~Wpadding+dilation~W(kernel_size~H~R1)+output_padding+1
        self.upsample_conv = upsample_block(channels, k, label_rate[0])

        self.upsample_rate, downsample_rate = label_rate
        assert downsample_rate == 1, "must be 1 to perform downsample only"
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.highway = highway  # Useless
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x, padding=None, mask_indices=None):
        # Assume x1 = (B, T, C) as input
        x = x.permute(0, 2, 1)
        residual_before_upsample = x
        x = self.upsample_conv(x)
        upsample_size = x.size(2)

        # conduct upsample
        if self.skip_connections:
            residual_upsample = torch.repeat_interleave(
                residual_before_upsample, self.upsample_rate, dim=2
            )
            upsample_size = min(upsample_size, residual_upsample.size(2))
            x = (
                x[..., :upsample_size] + residual_upsample[..., :upsample_size]
            ) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        x = x.permute(0, 2, 1)

        # process padding
        if padding is not None:
            padding = torch.repeat_interleave(padding, self.upsample_rate, dim=1)
            padding = padding[..., : x.size(1)]

        # process mask indices
        if mask_indices is not None:
            mask_indices = torch.repeat_interleave(
                mask_indices, self.upsample_rate, dim=1
            )
            mask_indices = mask_indices[..., : x.size(1)]
        return x, padding, mask_indices
