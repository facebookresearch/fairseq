# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import math
from typing import Optional, Tuple
from omegaconf import II
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GumbelVectorQuantizer,
    KmeansVectorQuantizer,
    TransposeLast,
)
from fairseq.tasks import FairseqTask
from fairseq.utils import buffered_arange


logger = logging.getLogger(__name__)


AGGREGATOR_CHOICES = ChoiceEnum(["cnn", "gru"])
PROJECT_FEATURES_CHOICES = ChoiceEnum(["none", "same", "new"])
ACTIVATION_CHOICES = ChoiceEnum(["relu", "gelu"])
VQ_TYPE_CHOICES = ChoiceEnum(["none", "gumbel", "kmeans"])


@dataclass
class Wav2VecConfig(FairseqDataclass):
    prediction_steps: int = field(
        default=12, metadata={"help": "number of steps ahead to predict"}
    )
    sample_distance: Optional[int] = field(
        default=None,
        metadata={
            "help": "sample distance from target. does not work properly with cross-sampling"
        },
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "num of cross sampled negatives"}
    )
    num_negatives: int = field(
        default=10, metadata={"help": "num of cross sampled negatives"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)]",
        metadata={
            "help": "convolutional feature extraction layers [(dim, kernel_size, stride), ...]"
        },
    )
    conv_aggregator_layers: str = field(
        default="[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]",
        metadata={
            "help": "convolutional aggregator layers [(dim, kernel_size, stride), ...]"
        },
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout to apply within the model"}
    )
    dropout_features: float = field(
        default=0.0, metadata={"help": "dropout to apply to the features"}
    )
    dropout_agg: float = field(
        default=0.0, metadata={"help": "dropout to apply after aggregation step"}
    )
    aggregator: AGGREGATOR_CHOICES = field(
        default="cnn", metadata={"help": "type of aggregator to use"}
    )
    gru_dim: int = field(default=512, metadata={"help": "GRU dimensionality"})
    no_conv_bias: bool = field(
        default=False, metadata={"help": "if set, does not learn bias for conv layers"}
    )
    agg_zero_pad: bool = field(
        default=False,
        metadata={"help": "if set, zero pads in aggregator instead of repl pad"},
    )
    skip_connections_feat: bool = field(
        default=False,
        metadata={"help": "if set, adds skip connections to the feature extractor"},
    )
    skip_connections_agg: bool = field(
        default=True,
        metadata={"help": "if set, adds skip connections to the aggregator"},
    )
    residual_scale: float = field(
        default=0.5, metadata={"help": "scales residual by sqrt(value)"}
    )
    log_compression: bool = field(
        default=True,
        metadata={"help": "if set, adds a log compression to feature extractor"},
    )
    balanced_classes: bool = field(
        default=False,
        metadata={"help": "if set, loss is scaled to balance for number of negatives"},
    )
    project_features: PROJECT_FEATURES_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, features are projected using the (same or new) aggregator"
        },
    )
    non_affine_group_norm: bool = field(
        default=False, metadata={"help": "if set, group norm is not affine"}
    )
    offset: str = field(
        default="auto",
        metadata={
            "help": "if set to 'auto', it is computed automatically from the receptive field, else set to int value"
        },
    )
    activation: ACTIVATION_CHOICES = field(
        default="relu",
        metadata={
            "help": "if set to 'auto', it is computed automatically from the receptive field, else set to int value"
        },
    )
    vq_type: VQ_TYPE_CHOICES = field(
        default="none", metadata={"help": "which type of quantizer to use"}
    )
    vq_vars: int = field(
        default=320,
        metadata={"help": "project to this many vector quantized variables per group"},
    )
    vq_groups: int = field(
        default=2, metadata={"help": "number of groups of latent variables"}
    )
    vq_dim: int = field(
        default=0,
        metadata={
            "help": "uses this dimensionality for quantized vectors. 0 to use model dim // groups"
        },
    )
    vq_depth: int = field(
        default=1, metadata={"help": "number of layers for vq weight projection"}
    )
    combine_groups: bool = field(
        default=False, metadata={"help": "if set, variables are shared among groups"}
    )
    vq_temp: Tuple[float, float, float] = field(
        default=(2.0, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling with gumbel softmax. should be a tuple of 3 values (start, end, decay)"
        },
    )
    vq_gamma: float = field(
        default=0.25,
        metadata={"help": "gamma parameter for kmeans style vector quantization"},
    )
    infonce: bool = II("criterion.infonce")


@register_model("wav2vec", dataclass=Wav2VecConfig)
class Wav2VecModel(BaseFairseqModel):
    @classmethod
    def build_model(cls, cfg: Wav2VecConfig, task: FairseqTask):
        """Build a new model instance."""

        model = Wav2VecModel(cfg)
        logger.info(model)
        return model

    def __init__(self, cfg: Wav2VecConfig):
        super().__init__()

        self.prediction_steps = cfg.prediction_steps
        offset = cfg.offset

        if cfg.activation == "relu":
            activation = nn.ReLU()
        elif cfg.activation == "gelu":
            activation = nn.GELU()
        else:
            raise Exception("unknown activation " + cfg.activation)

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            log_compression=cfg.log_compression,
            skip_connections=cfg.skip_connections_feat,
            residual_scale=cfg.residual_scale,
            non_affine_group_norm=cfg.non_affine_group_norm,
            activation=activation,
        )
        embed = feature_enc_layers[-1][0]

        self.vector_quantizer = None
        if cfg.vq_type == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=embed,
                num_vars=cfg.vq_vars,
                temp=cfg.vq_temp,
                groups=cfg.vq_groups,
                combine_groups=cfg.combine_groups,
                vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,
                time_first=False,
                activation=activation,
                weight_proj_depth=cfg.vq_depth,
                weight_proj_factor=2,
            )
        elif cfg.vq_type == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=embed,
                num_vars=cfg.vq_vars,
                groups=cfg.vq_groups,
                combine_groups=cfg.combine_groups,
                vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,
                time_first=False,
                gamma=cfg.vq_gamma,
            )
        else:
            assert (
                cfg.vq_type == "none" or cfg.vq_type is None
            ), "Unknown quantizer type"

        if cfg.offset == "auto":
            jin = 0
            rin = 0
            for _, k, stride in feature_enc_layers:
                if rin == 0:
                    rin = k
                rin = rin + (k - 1) * jin
                if jin == 0:
                    jin = stride
                else:
                    jin *= stride
            offset = math.ceil(rin / jin)

        offset = int(offset)

        def make_aggregator():
            if cfg.aggregator == "cnn":
                agg_layers = eval(cfg.conv_aggregator_layers)
                agg_dim = agg_layers[-1][0]
                feature_aggregator = ConvAggegator(
                    conv_layers=agg_layers,
                    embed=embed,
                    dropout=cfg.dropout,
                    skip_connections=cfg.skip_connections_agg,
                    residual_scale=cfg.residual_scale,
                    non_affine_group_norm=cfg.non_affine_group_norm,
                    conv_bias=not cfg.no_conv_bias,
                    zero_pad=cfg.agg_zero_pad,
                    activation=activation,
                )
            elif cfg.aggregator == "gru":
                agg_dim = cfg.gru_dim
                feature_aggregator = nn.Sequential(
                    TransposeLast(),
                    nn.GRU(
                        input_size=embed,
                        hidden_size=agg_dim,
                        num_layers=1,
                        dropout=cfg.dropout,
                    ),
                    TransposeLast(deconstruct_idx=0),
                )
            else:
                raise Exception("unknown aggregator type " + cfg.aggregator)

            return feature_aggregator, agg_dim

        self.feature_aggregator, agg_dim = make_aggregator()

        self.wav2vec_predictions = Wav2VecPredictionsModel(
            in_dim=agg_dim,
            out_dim=embed,
            prediction_steps=cfg.prediction_steps,
            n_negatives=cfg.num_negatives,
            cross_sample_negatives=cfg.cross_sample_negatives,
            sample_distance=cfg.sample_distance,
            dropout=cfg.dropout,
            offset=offset,
            balanced_classes=cfg.balanced_classes,
            infonce=cfg.infonce,
        )

        self.dropout_feats = nn.Dropout(p=cfg.dropout_features)
        self.dropout_agg = nn.Dropout(p=cfg.dropout_agg)

        if cfg.project_features == "none":
            self.project_features = None
        elif cfg.project_features == "same":
            self.project_features = self.feature_aggregator
        elif cfg.project_features == "new":
            self.project_features, _ = make_aggregator()

    def forward(self, source):
        result = {}

        features = self.feature_extractor(source)
        if self.vector_quantizer:
            q_res = self.vector_quantizer(features)
            features = q_res["x"]
            for k in q_res.keys():
                if k != "x":
                    result[k] = q_res[k]

        x = self.dropout_feats(features)
        x = self.feature_aggregator(x)
        x = self.dropout_agg(x)

        if self.project_features is not None:
            features = self.project_features(features)
        x, targets = self.wav2vec_predictions(x, features)
        result["cpc_logits"] = x
        result["cpc_targets"] = targets

        return result

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

    def max_positions(self):
        """Maximum length supported by the model."""
        return sys.maxsize

    def get_logits(self, net_output):
        logits = net_output["cpc_logits"]
        return logits

    def get_targets(self, sample, net_output):
        t = net_output["cpc_targets"]
        if isinstance(t, tuple):
            t = t[0]
        return t.contiguous()

    def get_target_weights(self, targets, net_output):
        targets = net_output["cpc_targets"]
        if isinstance(targets, tuple) and targets[-1] is not None:
            return targets[-1]
        return None

    def get_extra_losses(self, net_output):
        loss = None
        if "prob_perplexity" in net_output:
            loss = net_output["num_vars"] - net_output["prob_perplexity"]
        elif "kmeans_loss" in net_output:
            loss = net_output["kmeans_loss"]

        return loss


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers,
        dropout,
        log_compression,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


class ConvAggegator(nn.Module):
    def __init__(
        self,
        conv_layers,
        embed,
        dropout,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        conv_bias,
        zero_pad,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka

            pad = (
                ZeroPad1d(ka + kb, 0) if zero_pad else nn.ReplicationPad1d((ka + kb, 0))
            )

            return nn.Sequential(
                pad,
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                norm_block(False, n_out, affine=not non_affine_group_norm),
                activation,
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


class Wav2VecPredictionsModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        prediction_steps,
        n_negatives,
        cross_sample_negatives,
        sample_distance,
        dropout,
        offset,
        balanced_classes,
        infonce,
    ):
        super().__init__()

        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = sample_distance
        self.project_to_steps = nn.ConvTranspose2d(
            in_dim, out_dim, (1, prediction_steps)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes
        self.infonce = infonce

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        cross_high = tsz * bsz
        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1

        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * tsz),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, self.n_negatives + self.cross_sample_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT

        return negs

    def forward(self, x, y):

        x = x.unsqueeze(-1)
        x = self.project_to_steps(x)  # BxCxTxS
        x = self.dropout(x)

        negatives = self.sample_negatives(y)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)  # Copies x B x C x T

        copies = targets.size(0)
        bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)

        predictions = x.new(
            bsz * copies * (tsz - self.offset + 1) * steps
            - ((steps + 1) * steps // 2) * copies * bsz
        )
        if self.infonce:
            labels = predictions.new_full(
                (predictions.shape[0] // copies,), 0, dtype=torch.long
            )
        else:
            labels = torch.zeros_like(predictions)
        weights = (
            torch.full_like(labels, 1 / self.n_negatives)
            if self.balanced_classes and not self.infonce
            else None
        )

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            if self.infonce:
                predictions[start:end] = torch.einsum(
                    "bct,nbct->tbn", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
            else:
                pos_num = (end - start) // copies
                predictions[start:end] = torch.einsum(
                    "bct,nbct->nbt", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
                labels[start : start + pos_num] = 1.0
                if weights is not None:
                    weights[start : start + pos_num] = 1.0
            start = end
        assert end == predictions.numel(), "{} != {}".format(end, predictions.numel())

        if self.infonce:
            predictions = predictions.view(-1, copies)
        else:
            if weights is not None:
                labels = (labels, weights)

        return predictions, labels
