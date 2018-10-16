# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.modules import ElmoTokenEmbedder
from . import (
    BaseFairseqModel, register_model, register_model_architecture,
)

from fairseq import options
from fairseq import utils

from fairseq.models.transformer import (
    Embedding, LayerNorm, Linear, PositionalEmbedding,
)


@register_model('sentence_classifier')
class SentenceClassifier(BaseFairseqModel):
    def __init__(self, args, embedding):
        super().__init__()

        self.embedding = embedding
        self.linear = nn.Linear(args.model_dim, args.num_labels)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--elmo-path', metavar='PATH', help='path to elmo model')
        parser.add_argument('--model-dim', type=int, metavar='N', help='decoder input dimension')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        dictionary = task.dictionary

        if args.elmo_path is not None:
            task = LanguageModelingTask(args, dictionary, dictionary)
            models, _ = utils.load_ensemble_for_inference([args.elmo_path], task, {'remove_head': True})
            assert len(models) == 1, 'ensembles are currently not supported for elmo embeddings'

            embedding = ElmoTokenEmbedder(
                models[0],
                dictionary.eos(),
                dictionary.pad(),
                combine_tower_states=True,
                projection_dim=args.model_dim,
                add_final_predictive=True,
                add_final_context=True,
                weights_dropout=0,
                tune_lm=False,
                apply_softmax=True,
                layer_norm=False,
                affine_layer_norm=False,
                channelwise_weights=False,
                scaled_sigmoid=False,
                individual_norms=False,
                channelwise_norm=False,
                init_gamma=1.0,
                ltn=True,
                ltn_dims=3,
                train_gamma=True,
            )
        else:
            embedding = nn.Embedding(len(dictionary), args.model_dim, dictionary.pad())

        return SentenceClassifier(args, embedding)

    def forward(self, src_tokens, src_lengths):
        x = self.embedding(src_tokens)
        x = self.linear(x)
        x = x.mean(dim=1)
        return x

    # def get_normalized_probs(self, net_output, log_probs, sample=None):
    #     pass


@register_model_architecture('sentence_classifier', 'sentence_classifier')
def base_architecture(args):
    args.model_dim = getattr(args, 'model_dim', 2048)
