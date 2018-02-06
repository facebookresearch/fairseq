# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


class LabelSmoothedNLLLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, target, eps, padding_idx, weights, reduce=True):
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
        if reduce:
            return input.new([grad_input.view(-1).dot(input.view(-1))])
        else:
            return grad_input * input

    @staticmethod
    def backward(ctx, grad):
        return utils.volatile_variable(ctx.grad_input) * grad, None, None, None, None, None


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, src_dict, dst_dict):
        super().__init__(args, src_dict, dst_dict)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample['target'].view(-1)
        loss = LabelSmoothedNLLLoss.apply(lprobs, target, self.eps, self.padding_idx, None, reduce)
        nll_loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data[0] if reduce else loss.data,
            'nll_loss': nll_loss.data[0] if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
        }
