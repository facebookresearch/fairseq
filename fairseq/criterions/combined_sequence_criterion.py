# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math

from .fairseq_sequence_criterion import FairseqSequenceCriterion

class CombinedSequenceCriterion(FairseqSequenceCriterion):

    def __init__(self, args, dst_dict, token_criterion, sequence_criterion, alpha):
        super().__init__(args, dst_dict)
        self.token_criterion = token_criterion
        self.sequence_criterion = sequence_criterion
        self.alpha = alpha

    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        # compute token-level loss (unnormalized)
        sample['token_criterion_out'] = self.token_criterion(model, sample)

        # then prepare sample for sequence-level criterion
        return self.sequence_criterion.prepare_sample_and_hypotheses(model, sample, hypos)

    def get_net_output(self, model, sample):
        return self.sequence_criterion.get_net_output(model, sample)

    def sequence_forward(self, net_output, model, sample):
        """Compute the sequence-level loss for the given hypotheses.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # get token-level loss
        tok_loss, tok_sample_size, _ = sample['token_criterion_out']

        # compute sequence-level loss
        seq_loss, seq_sample_size, seq_logging_output = \
                self.sequence_criterion.sequence_forward(net_output, model, sample)

        # First normalize the loss using the current sample's size.
        # Later, normalize again by the number of replicas.

        loss = self.alpha  * tok_loss / tok_sample_size + \
                   (1 - self.alpha) * seq_loss.sum() / seq_sample_size

        loss = loss.sum()
        sample_size = 1  # normalize gradients by the number of replicas

        seq_logging_output.update({
            'loss': loss.data[0],
            'sample_size': sample_size,
            'tok_loss': tok_loss.data.sum(),
            'tok_sample_size': tok_sample_size,
            'seq_loss': seq_loss.data.sum(),
            'seq_sample_size': seq_sample_size,
        })
        return loss, sample_size, seq_logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        tok_sample_size = sum(log.get('tok_sample_size', 0) for log in logging_outputs)
        seq_sample_size = sum(log.get('seq_sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size ,
            'tok_loss': sum(log.get('tok_loss', 0) for log in logging_outputs) / tok_sample_size / math.log(2),
            'seq_loss': sum(log.get('seq_loss', 0) for log in logging_outputs) / seq_sample_size,
        }
