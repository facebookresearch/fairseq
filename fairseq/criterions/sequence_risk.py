# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqSequenceCriterion, register_criterion


@register_criterion('sequence_risk')
class SequenceRiskCriterion(FairseqSequenceCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        from fairseq.tasks.translation_struct import TranslationStructuredPredictionTask
        if not isinstance(task, TranslationStructuredPredictionTask):
            raise Exception(
                'sequence_risk criterion requires `--task=translation_struct`'
            )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        bsz = len(sample['hypos'])
        nhypos = len(sample['hypos'][0])

        # get costs for hypotheses using --seq-scorer (defaults to 1. - BLEU)
        costs = self.task.get_costs(sample)

        # generate a new sample from the given hypotheses
        new_sample = self.task.get_new_sample_for_hypotheses(sample)
        hypotheses = new_sample['target'].view(bsz, nhypos, -1, 1)
        hypolen = hypotheses.size(2)
        pad_mask = hypotheses.ne(self.task.target_dictionary.pad())
        lengths = pad_mask.sum(dim=2).float()

        net_output = model(**new_sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(bsz, nhypos, hypolen, -1)

        scores = lprobs.gather(3, hypotheses)
        scores *= pad_mask.float()
        avg_scores = scores.sum(dim=2) / lengths
        probs = F.softmax(avg_scores.exp(), dim=1).squeeze(-1)
        loss = (probs * costs).sum()

        sample_size = bsz
        logging_output = {
            'loss': utils.item(loss.data),
            'sum_cost': utils.item(costs.sum()),
            'num_cost': costs.numel(),
            'ntokens': sample['ntokens'],
            'nsentences': bsz,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'avg_cost': sum(log.get('sum_cost', 0) for log in logging_outputs) /
                sum(log.get('num_cost', 0) for log in logging_outputs),
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
