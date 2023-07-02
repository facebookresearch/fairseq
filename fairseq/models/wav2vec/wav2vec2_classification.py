# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES, Wav2Vec2Config
from fairseq.models.wav2vec.wav2vec2_asr import Embedding, Linear, Wav2VecEncoder, Wav2Vec2AsrConfig
from fairseq.tasks import FairseqTask

logging.basicConfig(level=logging.DEBUG)


@dataclass
class Wav2Vec2ClassificationConfig(Wav2Vec2AsrConfig):
    latent_embed_dim: Optional[int] = field(
        default=None, metadata={"help": "latent dim (encoder w2v -> latent -> class"}
    )
    pooling: str = field(
        default="first_token",
        metadata={"help": "pooling layer choices"},
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )


@register_model("wav2vec_classification", dataclass=Wav2Vec2ClassificationConfig)
class Wav2VecClassification(BaseFairseqModel):
    # TODO: Can be shared/merged with ASR model class as w2v_encoder params are common.
    def __init__(
        self,
        cfg: Wav2Vec2ClassificationConfig,
        w2v_encoder: BaseFairseqModel,
        pooling_layer,
    ):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.pooling_layer = pooling_layer

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2ClassificationConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, None)
        pooling_layer = get_pooling_layer(
            cfg,
            w2v_encoder.w2v_model.encoder.layers[-1].embedding_dim,
            len(task.target_dictionary),
            len(w2v_encoder.w2v_model.encoder.layers),
        )
        return cls(cfg, w2v_encoder, pooling_layer)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        return net_output

    def forward(self, **kwargs):
        encoder_out_dict = self.w2v_encoder(**kwargs)
        w2v_encoder_out = encoder_out_dict["encoder_out"]  # TxBxC
        w2v_encoder_padding_mask = encoder_out_dict["padding_mask"]  # BxT
        # w2v_encoder_layer_results = encoder_out_dict["layer_results"]
        return self.pooling_layer(
            last_layer_feats=w2v_encoder_out,
            padding_mask=w2v_encoder_padding_mask,
            # all_layer_feats=w2v_encoder_layer_results,
        )

    # def forward_latent(self, **kwargs):
    #     encoder_out_dict = self.w2v_encoder(**kwargs)
    #     w2v_encoder_out = encoder_out_dict["encoder_out"]
    #     w2v_encoder_padding_mask = encoder_out_dict["encoder_padding_mask"]
    #     w2v_encoder_layer_results = encoder_out_dict["layer_results"]
    #     return self.pooling_layer.forward_latent(
    #         last_layer_feats=w2v_encoder_out,
    #         padding_mask=w2v_encoder_padding_mask,
    #         all_layer_feats=w2v_encoder_layer_results,
    #     )


def get_pooling_layer(
    cfg: Wav2Vec2ClassificationConfig,
    encoder_embed_dim: int,
    num_targets: int,
    encoder_layers: int,
):
    assert cfg.pooling == 'mean'
    if cfg.pooling == "first_token":
        return FirstToken(cfg, encoder_embed_dim, num_targets)
    # elif cfg.pooling == "mean":
    #     return MeanPooling(cfg, encoder_embed_dim, num_targets)
    elif cfg.pooling == "mean":
        return MeanPoolingFast(cfg, encoder_embed_dim, num_targets)
    elif cfg.pooling == "mean_amsoftmax":
        return MeanPoolingFastAMSoftmax(cfg, encoder_embed_dim, num_targets)
    elif cfg.pooling == "max":
        return MaxPoolingFast(cfg, encoder_embed_dim, num_targets)
    elif cfg.pooling == "elmo":
        return LayerWeightedMeanPooling(
            cfg, encoder_embed_dim, num_targets, encoder_layers
        )
    else:
        raise NotImplementedError(f"{cfg.pooling} has not been implemented yet.")


class Pooling(nn.Module):
    def __init__(
        self,
        cfg: Wav2Vec2ClassificationConfig,
        encoder_embed_dim: int,
        num_targets: int,
    ):
        super().__init__()
        self.projection = Linear(encoder_embed_dim, num_targets)

    def forward(self, last_layer_feats, **kwargs):
        raise NotImplementedError()


class FirstToken(Pooling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, last_layer_feats, **kwargs):
        return self.projection(last_layer_feats[:, 0])


# class MeanPooling(Pooling):
#     def __init__(
#         self,
#         cfg: Wav2VecClassificationConfig,
#         encoder_embed_dim: int,
#         num_targets: int,
#         **kwargs,
#     ):
#         super().__init__(cfg, encoder_embed_dim, num_targets)
#         self.activation_fn = utils.get_activation_fn(cfg.activation_fn)
#         self.linear = Linear(encoder_embed_dim, encoder_embed_dim)

#     def forward(self, last_layer_feats, padding_mask, **kwargs):
#         # last_layer_feats: [BxTxD]
#         # padding_mask: [BxT]
#         last_layer_feats = self.linear(self.activation_fn(last_layer_feats))
#         input_lengths = (1 - padding_mask.long()).sum(-1)
#         pooled_feature_list = []
#         for i in range(len(last_layer_feats)):
#             length = input_lengths[i]
#             pooled_feature = torch.mean(last_layer_feats[i][:length], dim=0)
#             pooled_feature_list.append(pooled_feature)
#         return self.projection(torch.stack(pooled_feature_list))


def fn_mean(x, mask):
    """
    Args:
        x: TxBxD
        mask: BxT
    Return:
        y: BxD
    """
    if mask is not None:
        mask = mask.t()[:, :, None]
        return (x * mask).sum(0) / mask.sum(0)
    else:
        return x.sum(0) / x.shape[0]


class MeanPoolingFast(nn.Module):
    def __init__(
        self,
        cfg: Wav2Vec2ClassificationConfig,
        encoder_embed_dim: int,
        num_targets: int,
        **kwargs,
    ):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(cfg.activation_fn)
        self.latent_embed_dim = (
            cfg.latent_embed_dim
            if cfg.latent_embed_dim is not None
            else encoder_embed_dim
        )
        logging.debug(f"| {self.latent_embed_dim=}")
        self.linear = Linear(encoder_embed_dim, self.latent_embed_dim)
        self.projection = Linear(self.latent_embed_dim, num_targets)

    def forward(self, last_layer_feats, padding_mask, **kwargs):
        """
        Arguments
            features - [TxBxD] Acoustic feature with shape
            padding_mask - [BxT]     Padding Mask
        """
        if padding_mask is not None:
            feat_mask = (~padding_mask).to(last_layer_feats.dtype)
        else:
            feat_mask = None
        feat = self.linear(last_layer_feats)
        feat = fn_mean(feat, feat_mask)
        feat = self.activation_fn(feat)
        return self.projection(feat)

    def forward_latent(self, last_layer_feats, padding_mask, **kwargs):
        """
        Arguments
            features - [TxBxD] Acoustic feature with shape
            padding_mask - [BxT]     Padding Mask
        """
        if padding_mask is not None:
            feat_mask = (~padding_mask).to(last_layer_feats.dtype)
        else:
            feat_mask = None
        feat = self.linear(last_layer_feats)
        feat = fn_mean(feat, feat_mask)
        return feat


class MeanPoolingFastAMSoftmax(MeanPoolingFast):
    def __init__(
        self,
        cfg: Wav2Vec2ClassificationConfig,
        encoder_embed_dim: int,
        num_targets: int,
        **kwargs,
    ):
        super().__init__(cfg, encoder_embed_dim, num_targets, **kwargs)
        self.projection = Linear(self.latent_embed_dim, num_targets, bias=False)
        nn.init.xavier_normal_(self.projection.weight, gain=1)

    def forward(self, last_layer_feats, padding_mask, **kwargs):

        """
        Arguments
            features - [BxTxD] Acoustic feature with shape
            padding_mask - [BxT]     Padding Mask
        """
        feat_mask = (~padding_mask).to(last_layer_feats.dtype)  # T,B -> B,T
        feat = self.linear(last_layer_feats)  # B,T,D
        feat = fn_mean(feat, feat_mask)  # B,D
        feat = self.activation_fn(feat)
        # normalize feat
        feat_norm = F.normalize(feat, p=2, dim=-1)  # B,D
        weight_norm = F.normalize(self.projection.weight.t(), p=2, dim=-1)  # D,K
        cos_fw = feat_norm @ weight_norm
        return cos_fw


def fn_max(x, mask):
    """
    Args:
        x: TxBxD
        mask: BxT
    Return:
        y: BxD
    """
    mask = mask.t()[:, :, None].to(torch.bool)
    return x.masked_fill(~mask, -1e-8).max(0)[0]


class MaxPoolingFast(Pooling):
    def __init__(
        self,
        cfg: Wav2Vec2ClassificationConfig,
        encoder_embed_dim: int,
        num_targets: int,
        **kwargs,
    ):
        super().__init__(cfg, encoder_embed_dim, num_targets)
        self.activation_fn = utils.get_activation_fn(cfg.activation_fn)
        self.linear = Linear(encoder_embed_dim, encoder_embed_dim)

    def forward(self, last_layer_feats, padding_mask, **kwargs):

        """
        Arguments
            features - [TxBxD] Acoustic feature with shape
            padding_mask - [BxT]     Padding Mask
        """
        feat_mask = (~padding_mask).to(last_layer_feats.dtype)
        feat = self.linear(last_layer_feats)
        feat = fn_max(feat, feat_mask)
        feat = self.activation_fn(feat)
        return self.projection(feat)


class LayerWeightedMeanPooling(MeanPoolingFast):
    """Elmo-style weighted average representation."""

    def __init__(
        self,
        cfg: Wav2Vec2ClassificationConfig,
        encoder_embed_dim: int,
        num_targets: int,
        encoder_layers: int,
    ):
        super().__init__(cfg, encoder_embed_dim, num_targets)
        self.num_layers = encoder_layers
        self.weights = nn.Parameter(torch.ones(encoder_layers))

    def forward(self, last_layer_feats, padding_mask, all_layer_feats):
        # last_layer_feats: [BxTxD]
        # padding_mask: [BxT]
        if not self.training:
            msg = (
                f"Number of layers in input features = {len(all_layer_feats)}."
                f" Expected {self.num_layers} layers."
            )
            assert len(all_layer_feats) == self.num_layers, msg

        # Stack up all layers and reshape to (num_layers, features)
        all_layer_feats_stacked = torch.stack(all_layer_feats, dim=0)
        num_layers, *original_feat_shape = all_layer_feats_stacked.shape
        all_layer_feats_stacked_flat = all_layer_feats_stacked.view(num_layers, -1)

        # Weighted average
        normalized_weights = F.softmax(self.weights, dim=-1)
        weighted_avg_features = (
            normalized_weights.unsqueeze(-1) * all_layer_feats_stacked_flat
        ).sum(dim=0)
        weighted_avg_features = weighted_avg_features.view(*original_feat_shape)

        # Mean Pooling on weighted average features.
        return super().forward(weighted_avg_features, padding_mask)