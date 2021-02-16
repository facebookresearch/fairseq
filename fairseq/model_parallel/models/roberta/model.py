# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.model_parallel.models.transformer import ModelParallelTransformerEncoder
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import (
    roberta_base_architecture,
    roberta_prenorm_architecture,
    RobertaEncoder,
    RobertaModel,
)
from fairseq.modules import LayerNorm


try:
    from fairseq.model_parallel.megatron.mpu import (
        copy_to_model_parallel_region,
        gather_from_model_parallel_region,
        ColumnParallelLinear,
        VocabParallelEmbedding,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False

logger = logging.getLogger(__name__)


@register_model("model_parallel_roberta")
class ModelParallelRobertaModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        RobertaModel.add_args(parser)
        parser.add_argument(
            "--no-final-layer-norm",
            action="store_true",
            help=(
                "don't add final layernorm (only applicable when "
                "--encoder-normalize-before=True"
            ),
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        task.source_dictionary.pad_to_multiple_(args.model_parallel_size * 8)
        task.target_dictionary.pad_to_multiple_(args.model_parallel_size * 8)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        if getattr(args, "untie_weights_roberta", False):
            raise NotImplementedError(
                "--untie-weights-roberta is not supported in model parallel mode"
            )

        encoder = ModelParallelRobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ModelParallelRobertaClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )


class ModelParallelRobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = ColumnParallelLinear(embed_dim, embed_dim, gather_output=True)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)

        x = copy_to_model_parallel_region(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight)
        x = gather_from_model_parallel_region(x).contiguous()
        x = x + self.bias
        return x


class ModelParallelRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout
    ):
        super().__init__()
        self.dense = ColumnParallelLinear(input_dim, inner_dim, gather_output=True)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ModelParallelRobertaEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        assert not self.args.untie_weights_roberta

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return VocabParallelEmbedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        return ModelParallelTransformerEncoder(args, dictionary, embed_tokens)

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return ModelParallelRobertaLMHead(embed_dim, output_dim, activation_fn, weight)


@register_model_architecture("model_parallel_roberta", "model_parallel_roberta")
def base_architecture(args):
    args.no_final_layer_norm = getattr(args, "no_final_layer_norm", False)
    # model parallel RoBERTa defaults to "Pre-LN" formulation
    roberta_prenorm_architecture(args)


# earlier versions of model parallel RoBERTa removed the final layer norm
@register_model_architecture("model_parallel_roberta", "model_parallel_roberta_v1")
def model_parallel_roberta_v1_architecture(args):
    args.no_final_layer_norm = getattr(args, "no_final_layer_norm", True)
    base_architecture(args)


@register_model_architecture(
    "model_parallel_roberta", "model_parallel_roberta_postnorm"
)
def model_parallel_roberta_postnorm_architecture(args):
    # the original BERT/RoBERTa uses the "Post-LN" formulation
    roberta_base_architecture(args)


@register_model_architecture("model_parallel_roberta", "model_parallel_roberta_base")
def model_parallel_roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture("model_parallel_roberta", "model_parallel_roberta_large")
def model_parallel_roberta_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)
