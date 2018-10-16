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
    def __init__(self, args, dictionary):
        super().__init__()

        dim = 1024
        self.embedding = nn.Embedding(len(dictionary), dim, dictionary.pad())
        self.linear = nn.Linear(dim, args.num_labels)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        return SentenceClassifier(args, task.dictionary)

    def forward(self, src_tokens, src_lengths):
        x = self.embedding(src_tokens)
        x = self.linear(x)
        x = x.mean(dim=1)
        return x

    # def get_normalized_probs(self, net_output, log_probs, sample=None):
    #     pass


@register_model_architecture('sentence_classifier', 'sentence_classifier')
def base_architecture(args):
    pass
