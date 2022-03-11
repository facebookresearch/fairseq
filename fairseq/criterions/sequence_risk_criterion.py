# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .fairseq_sequence_criterion import FairseqSequenceCriterion


class SequenceRiskCriterion(FairseqSequenceCriterion):

    def __init__(self, args, dst_dict):
        super().__init__(args, dst_dict)
        self.scale_scores = args.seq_risk_normbleu

    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        # compute BLEU cost for each hypothesis
        hypos = self.add_bleu_to_hypotheses(sample, hypos)
        if self.scale_scores:
            def minmax(iterable):
                _min = min(iterable)
                res = [x - _min for x in iterable]
                _max = max(res)
                return [x / (1e-6 +_max) for x in res]
            scale_scores = minmax
        else:
            scale_scores = lambda x : x

        cost = torch.FloatTensor([
            scale_scores([100 - h['bleu'] for h in hypos_i])
            for hypos_i in hypos
        ])

        sample['cost'] = Variable(cost, requires_grad=False)
        return sample, hypos

    def sequence_forward(self, net_output, model, sample):
        """Compute the sequence-level loss for the given hypotheses.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        scores = self.get_hypothesis_scores(net_output, sample)
        lengths = self.get_hypothesis_lengths(net_output, sample)
        avg_scores = scores.sum(2) / lengths
        probs = F.softmax(avg_scores.exp_())
        loss = (probs * sample['cost'].type_as(probs)).sum()
        sample_size = net_output.size(0)  # bsz
        logging_output = {
            'loss': loss.data[0],
            'sample_size': sample_size,
            'sum_cost': sample['cost'].sum(),
            'num_cost': sample['cost'].numel(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'avg_cost': sum(log.get('sum_cost', 0) for log in logging_outputs) /
                sum(log.get('num_cost', 0) for log in logging_outputs),
        }