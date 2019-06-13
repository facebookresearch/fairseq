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

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--normalize-costs', action='store_true',
                            help='normalize costs within each hypothesis')
        # fmt: on

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

        if self.args.normalize_costs:
            unnormalized_costs = costs.clone()
            max_costs = costs.max(dim=1, keepdim=True)[0]
            min_costs = costs.min(dim=1, keepdim=True)[0]
            costs = (costs - min_costs) / (max_costs - min_costs).clamp_(min=1e-6)
        else:
            unnormalized_costs = None

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
        probs = F.softmax(avg_scores, dim=1).squeeze(-1)
        loss = (probs * costs).sum()

        sample_size = bsz
        assert bsz == utils.item(costs.size(dim=0))
        logging_output = {
            'loss': utils.item(loss.data),
            'num_cost': costs.numel(),
            'ntokens': sample['ntokens'],
            'nsentences': bsz,
            'sample_size': sample_size,
        }

        def add_cost_stats(costs, prefix=''):
            logging_output.update({
                prefix + 'sum_cost': utils.item(costs.sum()),
                prefix + 'min_cost': utils.item(costs.min(dim=1)[0].sum()),
                prefix + 'cost_at_1': utils.item(costs[:, 0].sum()),
            })

        add_cost_stats(costs)
        if unnormalized_costs is not None:
            add_cost_stats(unnormalized_costs, 'unnormalized_')

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        num_costs = sum(log.get('num_cost', 0) for log in logging_outputs)
        agg_outputs = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }

        def add_cost_stats(prefix=''):
            agg_outputs.update({
                prefix + 'avg_cost': sum(log.get(prefix + 'sum_cost', 0) for log in logging_outputs) / num_costs,
                prefix + 'min_cost': sum(log.get(prefix + 'min_cost', 0) for log in logging_outputs) / nsentences,
                prefix + 'cost_at_1': sum(log.get(prefix + 'cost_at_1', 0) for log in logging_outputs) / nsentences,
            })

        add_cost_stats()
        if any('unnormalized_sum_cost' in log for log in logging_outputs):
            add_cost_stats('unnormalized_')

        return agg_outputs
