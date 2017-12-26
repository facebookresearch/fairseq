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


class LabelSmoothedNLLLoss(torch.autograd.Function):

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

    def __init__(self, args, dst_dict, weights=None):
        super().__init__(args, dst_dict)
        self.eps = args.label_smoothing
        self.weights = weights

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        input = F.log_softmax(net_output.view(-1, net_output.size(-1)), dim=1)
        target = sample['target'].view(-1)
        loss = LabelSmoothedNLLLoss.apply(input, target, self.eps, self.padding_idx, self.weights)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data[0],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
        }
