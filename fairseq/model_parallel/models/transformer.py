# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
)

from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
)

from fairseq.model_parallel.modules import (
    ModelParallelTransformerDecoderLayer,
    ModelParallelTransformerEncoderLayer,
)

try:
    from fairseq.model_parallel.megatron.mpu import (
        copy_to_model_parallel_region,
        gather_from_model_parallel_region,
        VocabParallelEmbedding,
    )
    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


logger = logging.getLogger(__name__)


@register_model('model_parallel_transformer')
class ModelParallelTransformerModel(TransformerModel):
    """
    Model parallel Transformer model.
    """
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        if not has_megatron_submodule:
            raise ImportError(
                '\n\nPlease install the megatron submodule:'
                '\n\n  git submodule update --init '
                'fairseq/model_parallel/megatron'
            )
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        def _vocab_init(tensor, **kwargs):
            nn.init.normal_(tensor, mean=0, std=num_embeddings ** -0.5)
            nn.init.constant_(tensor[1], 0)
        emb = VocabParallelEmbedding(num_embeddings, embed_dim, padding_idx, init_method=_vocab_init)
        # if provided, load from preloaded dictionaries
        if path:
            raise NotImplementedError("Loading of embedding from path is not supported for model parallel")
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ModelParallelTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ModelParallelTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )


class ModelParallelTransformerEncoder(TransformerEncoder):
    """
    Model parallel Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`ModelParallelTransformerEncoderLayer`.
    """

    def build_encoder_layer(self, args):
        return ModelParallelTransformerEncoderLayer(args)


class ModelParallelTransformerDecoder(TransformerDecoder):
    """
    Model Parallel Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`ModelParallelTransformerDecoderLayer`.
    """

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return ModelParallelTransformerDecoderLayer(args, no_encoder_attn)

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        features = copy_to_model_parallel_region(features)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(features, self.embed_tokens.weight)
        else:
            x = F.linear(features, self.embed_out)

        if getattr(self.args, 'criterion') != 'vocab_parallel_cross_entropy':
            x = gather_from_model_parallel_region(x).contiguous()
        return x
