# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import torch
from torch.autograd.variable import Variable
import torch.nn.functional as F

from .fairseq_criterion import FairseqCriterion


class LabelSmoothedCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, target, eps, padding_idx, weights):
        grad_input = input.new(input.size()).zero_()
        target = target.view(target.size(0), 1)
        grad_input = grad_input.scatter_(grad_input.dim() - 1, target, eps - 1)

        norm = grad_input.size(-1)
        if weights is not None:
            norm = weights.sum()
            grad_input.mul(weights.view(1, weights.size(0)).expand_as(grad_input))

        if padding_idx is not None:
            norm -= 1 if weights is None else weights[padding_idx]
            grad_input.select(grad_input.dim() - 1, padding_idx).fill_(0)

        grad_input = grad_input.add(-eps / norm)

        ctx.grad_input = grad_input
        return input.new([grad_input.view(-1).dot(input.view(-1))])

    @staticmethod
    def backward(ctx, grad):
        return Variable(ctx.grad_input, volatile=True) * grad, None, None, None, None


class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, eps, padding_idx=None, weights=None):
        super().__init__()
        self.eps = eps
        self.padding_idx = padding_idx
        self.weights = weights

    def grad_denom(self, samples):
        return sum(s['ntokens'] if s else 0 for s in samples)

    def forward(self, model, sample, grad_denom):
        net_output = model(**sample['net_input'])
        input = F.log_softmax(net_output.view(-1, net_output.size(-1)))
        target = sample['target'].view(-1)
        loss = LabelSmoothedCrossEntropy.apply(input, target, self.eps, self.padding_idx, self.weights)
        return {
            'loss': loss / grad_denom,
        }

    def aggregate(self, loss_dicts):
        return {
            'loss': sum(l['loss'].data[0] for l in loss_dicts if 'loss' in l) / math.log(2),
        }
