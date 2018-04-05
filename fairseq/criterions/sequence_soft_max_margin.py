# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import operator
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .fairseq_sequence_criterion import FairseqSequenceCriterion


class SequenceSoftMaxMarginCriterion(FairseqSequenceCriterion):

    def __init__(self, args, dst_dict):
        super().__init__(args, dst_dict)
        self.scale = args.seq_margin_cost_scale_factor

    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        # compute BLEU for each hypothesis
        hypos = self.add_bleu_to_hypotheses(sample, hypos)
        norm_bleu = torch.FloatTensor([
            [ self.scale * h['bleu'] / 100 for h in hypos_i]
            for hypos_i in hypos
        ])
        sample['norm_bleu'] = Variable(norm_bleu, requires_grad=False)
        # for each sentence, find the hypothesis with the maximum BLEU and set
        # it as the "target" hypothesis
        max_index = [
            max(enumerate(x['bleu'] for x in h), key=operator.itemgetter(1))[0]
            for h in hypos
        ]
        sample['target_hypo_idx'] = Variable(sample['target'].data.new(max_index), requires_grad=False)
        return sample, hypos

    def get_net_output(self, model, sample):
        """Return model outputs as log probabilities."""
        net_output = model(**sample['net_input'])
        return net_output.view(
            sample['bsz'], sample['num_hypos_per_batch'], -1, net_output.size(1))

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
        avg_scores = avg_scores - sample['norm_bleu'].type_as(avg_scores)
        loss = F.cross_entropy(avg_scores, sample['target_hypo_idx'], size_average=False)
        sample_size = net_output.size(0)  # bsz
        logging_output = {
            'loss': loss.data[0],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def aggregate(self, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
        }

