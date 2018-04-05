# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import operator
from torch.autograd import Variable
import torch.nn.functional as F
from .fairseq_sequence_criterion import FairseqSequenceCriterion


class SequenceCrossEntropyCriterion(FairseqSequenceCriterion):

    def __init__(self, args, dst_dict):
        super().__init__(args, dst_dict)

    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        # compute BLEU for each hypothesis
        hypos = self.add_bleu_to_hypotheses(sample, hypos)

        # for each sentence, find the hypothesis with the maximum BLEU and set
        # it as the "target" hypothesis
        max_index = [
            max(enumerate(x['bleu'] for x in h), key=operator.itemgetter(1))[0]
            for h in hypos
        ]
        sample['target_hypo_idx'] = Variable(sample['target'].data.new(max_index), requires_grad=False)
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
        loss = F.cross_entropy(avg_scores, sample['target_hypo_idx'], size_average=False)
        sample_size = net_output.size(0)  # bsz
        logging_output = {
            'loss': loss.data[0],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Compute the gradient denominator for a set of sample sizes."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
        }
