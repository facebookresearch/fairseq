# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn

from fairseq.tasks.fb_bert import BertTask
from . import (
    BaseFairseqModel, register_model, register_model_architecture,
)

from fairseq import utils

@register_model('finetuning_sentence_classifier')
class FinetuningSentenceClassifier(BaseFairseqModel):
    def __init__(self, args, pretrain_model):
        super().__init__()

        self.pretrain_model = pretrain_model
        self.last_dropout = nn.Dropout(args.last_dropout)
        self.proj = nn.Linear(args.model_dim, args.num_label)
        self.reset_parameters()

    def reset_parameters(self):
        self.proj.weight.data.normal_(mean=0.0, std=0.02)
        if self.proj.bias is not None:
            self.proj.bias.data.zero_()

    def forward(self, sentence, segment_labels):
        _, x = self.pretrain_model(sentence, segment_labels=segment_labels, apply_mask=False)
        if isinstance(x, list):
            x = x[0]
        x = self.last_dropout(x)
        x = self.proj(x)
        return x

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--bert-path', metavar='PATH', help='path to elmo model')
        parser.add_argument('--model-dim', type=int, metavar='N', help='decoder input dimension')
        parser.add_argument('--last-dropout', type=float, metavar='D', help='dropout before projection')
        parser.add_argument('--num-label', type=int, metavar='N', help='num of label class')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture_ft(args)

        dictionary = task.dictionary

        assert args.bert_path is not None
        args.short_seq_prob = 0.0
        task = BertTask(args, dictionary)
        models, _ = utils.load_ensemble_for_inference(
            [args.bert_path], task,
            {'remove_head': True})
        assert len(models) == 1, 'ensembles are currently not supported for elmo embeddings'

        return FinetuningSentenceClassifier(args, models[0])

@register_model_architecture('finetuning_sentence_classifier', 'finetuning_sentence_classifier')
def base_architecture_ft(args):
    args.model_dim = getattr(args, 'model_dim', 768)
    args.last_dropout = getattr(args, 'last_dropout', 0.1)

