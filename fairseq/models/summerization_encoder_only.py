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

from fairseq import options
from fairseq import utils


from fairseq.tasks.fb_bert import BertTask
from . import (
    BaseFairseqModel, register_model, register_model_architecture, FairseqIncrementalDecoder
)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


@register_model('ft_summerization')
class FTSummerization(BaseFairseqModel):

    def __init__(self, args, pretrain_model):
        super().__init__()
        self.pretrain_model = pretrain_model

    def forward(self, source, segment):
        enc_mask, dec_mask = self.generate_mask(segment)
        x, _ = self.pretrain_model(source, segment, apply_mask=False, mask=(enc_mask, dec_mask))
        return x

    def max_decoder_positions(self):
        return self.pretrain_model.decoder.max_positions

    def generate_mask(self, segment):
        segment = torch.cat([segment.new(segment.size(0), 1).fill_(0), segment], dim=-1)
        doc_mask = segment.eq(0)
        bsz, dim = segment.size()
        mask = utils.fill_with_neg_inf(segment.new(dim, dim))
        enc_mask, dec_mask = [], []
        for batch in range(bsz):
            enc = torch.triu(mask.clone(), 1)
            enc[doc_mask[batch].expand_as(enc).byte()] = 0
            dec = torch.triu(mask.clone(), 0)
            dec[doc_mask[batch].expand_as(dec).byte()] = 0
            enc_mask.append(enc)
            dec_mask.append(dec)
        return torch.stack(enc_mask, 0), torch.stack(dec_mask, 0)


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--bert-path', metavar='PATH', help='path to elmo model')
        parser.add_argument('--model-dim', type=int, metavar='N', help='decoder input dimension')
        parser.add_argument('--last-dropout', type=float, metavar='D', help='dropout before projection')
        parser.add_argument('--model-dropout', type=float, metavar='D', help='lm dropout')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='lm dropout')
        parser.add_argument('--relu-dropout', type=float, metavar='D', help='lm dropout')
        parser.add_argument('--proj-unk', action='store_true', help='if true, also includes unk emb in projection')
        parser.add_argument('--layer-norm', action='store_true', help='if true, does non affine layer norm before proj')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 512
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 512

        dictionary = task.source_dictionary
        assert args.bert_path is not None
        args.short_seq_prob = 0.0
        task = BertTask(args, dictionary)
        models, _ = utils.load_ensemble_for_inference([args.bert_path], task, {'save_masks' : False})
        assert len(models) == 1, 'ensembles are currently not supported for elmo embeddings'
        model = models[0]
        return FTSummerization(args, model)

@register_model_architecture('ft_summerization', 'ft_summerization')
def base_architecture(args):
    args.model_dim = getattr(args, 'model_dim', 768)
