# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.init import xavier_uniform_, xavier_normal_

from fairseq.models import register_model, BaseFairseqModel
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.models.wav2vec.wav2vec2 import *
from fairseq.modules.conformer_layer import ConformerEncoderLayer

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])
LAYER_TYPE_CHOICES = ChoiceEnum(["transformer", "conformer"])

@dataclass
class RandomProjectionConfig(Wav2Vec2Config):
    xavier_type: str = field(
        default="normal",
        metadata={
            "help": "xavier init type. options: uniform, normal"
        },
    )
    fbank_dim: int = field(
        default=80,
        metadata={
            "help": "fbank dim"
        },
    )
    fbank_shift: int= field(
        default=10,
        metadata={
            "help": "fbank shift stride"
        },
    )
    codebook_vocab_size: int = field(
        default= 8192,
        metadata={
            "help": "the vocab_size of codebook"
        }
    )
    codebook_dim: int = field(
        default= 16,
        metadata={
            "help": "the dim of codebook"
        }
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "the conv shape [(dim,kernel_size,stride)]"
        }
    )
    encoder_embed_dim: int =field(
        default = 1024,
        metadata={
            "input embedding dimenson of the conformer"
        }
    )
    encoder_ffn_embed_dim: int = field( # I can not find it in the paper...# maybe should use conformer paper
        default = 1024,
        metadata= {
            "FFN layer dimension"
        }
    )
    dropout_ratio: float = field(
        default = 0.0,
        metadata={
            "dropout ratio"
        } 
    )
    batch_size: int = field(
        default = 2048,
        metadata= {
            "batch size(fixed??)"
        }
    )
    encoder_layers: int = field(
        default = 24,
        metadata = {
            "encoder layers of conformer"
        }
    )
    encoder_depthwise_conv_kernel_size: int = field(
        default = 5,
        metadata = {
            "encoder depthwise conv kernel size"
        }
    )


@register_model("random_projection",dataclass = RandomProjectionConfig)
class RandomProjectionModel(BaseFairseqModel):
    def __init__(self, cfg:RandomProjectionConfig):
        super().__init__()
        self.cfg = cfg

        # conv_layer
        self.feature_enc_layers = eval(cfg.conv_feature_layers) # In the paper, there are 2 conv layers

        # dimension settings
        self.extractor_embed = self.feature_enc_layers[-1][0] # default: 512
        self.embedding_dim = cfg.encoder_embed_dim # input embedding dimenson of the conformer
        self.encoder_ffn_embed_dim = cfg.encoder_ffn_embed_dim # FFN layer dimension
        self.fbank_dim = cfg.fbank_dim # default: 80
        self.fbank_shift = cfg.fbank_shift # default: 10
        self.codebook_vocab_size = cfg.codebook_vocab_size # default: 8192
        self.codebook_dim = cfg.codebook_dim # default: 16
        self.dropout_ratio = cfg.dropout_ratio # default: 0.1
        self.feature_grad_mult = cfg.feature_grad_mult # default: 1.0

        # google usually use fbank, we use conv_feature extractor instead
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim)

        # mask settings
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        # mask prob settings
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        # dropout & layernorm
        self.dropout_input = nn.Dropout(cfg.dropout_input) # TODO
        self.layer_norm = nn.LayerNorm(self.extractor_embed)
        
        # conformer
        self.tgt_layer = cfg.encoder_layers
        self.encoder = nn.ModuleList(
            [ConformerEncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=cfg.encoder_ffn_embed_dim,
                attention_heads=cfg.encoder_attention_heads,
                dropout=cfg.dropout,
                use_fp16=True,
                depthwise_conv_kernel_size=cfg.encoder_depthwise_conv_kernel_size,
            ) for _ in range(cfg.encoder_layers)]
        )
        self.final_proj = nn.Linear(in_features=cfg.encoder_ffn_embed_dim, out_features=cfg.codebook_vocab_size)

        # random projection
        self.random_proj = nn.Parameter(torch.empty(self.embedding_dim, self.codebook_dim), requires_grad=False)
        self.codebook = nn.Parameter(torch.empty(self.codebook_vocab_size, self.codebook_dim), requires_grad=False)
        if(cfg.xavier_type == "normal"):
            self.random_proj = xavier_normal_(self.random_proj)
            self.codebook = xavier_normal_(self.codebook)
        elif(cfg.xavier_type == "uniform"):
            self.random_proj = xavier_uniform_(self.random_proj)
            self.codebook = xavier_uniform_(self.codebook)
        else:
            raise Exception(
                f"{cfg.xavier_type} is incorrect. Optional xavier types: normal, uniform"
            )

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        
        

    @classmethod
    def build_model(cls, cfg:RandomProjectionConfig,task=None):
        return cls(cfg)

    # some special function for random_projection
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

    # the function is same as data2vec
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
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
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
        features = source # [batch,length]

        # we use conv_feature extractor instead of fbank
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features) # [batch,channel,frame]
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        # in wav2vec, hubert, data2vec, we all use the layer_norm
        features = features.transpose(1, 2) # [batch,frame,channel]
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        # to get the code_books using unmasked_features
        with torch.no_grad():
            proj_vector = torch.matmul(unmasked_features,self.random_proj)
            proj_vector = torch.unsqueeze(proj_vector,2)
            norm2 = torch.squeeze(torch.sum(torch.sub(self.codebook, proj_vector) ** 2, dim=-1))
            labels = torch.argmin(norm2, dim=-1)

        # padding mask(used in data2vec)
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

        # post extract projection
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features) # [batch,frame,channel(512)] -> [batch,frame,channel(768)]

        # features dropout
        features = self.dropout_input(features)
        
        # masking
        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
        else:
            x = features
            mask_indices = None

        # encoder: using the conformer
        layer_results = []
        r = None # the last layer output
        position_emb = None
        for i, layer in enumerate(self.encoder):
            x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    position_emb=position_emb,
                )
            layer_results.append((x, z))

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        result = {
            "losses": {},
        }

        x = x[mask_indices] # only use masked part
        labels = labels[mask_indices] # only use masked part
        x = self.final_proj(x)

        sz = x.size(-1) # 768

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        # x = self.dropout_output(x) # [batch,frame,channel(768)]
        x = self.final_proj(x) # [batch,frame,channel(768)] -> [batch,frame,vocab_size(8192)]
        x = x.transpose(1, 2) # [batch,vocab_size,frame]
        
        # loss
        loss = self.criterion(x, labels) # labels: [batch,frame]

        result["losses"]["regression"] = loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        with torch.no_grad():
            result["target_var"] = self.compute_var(labels)
            result["pred_var"] = self.compute_var(x.float())

        return result
        
# other submodels