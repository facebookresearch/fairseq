# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math
import torch.nn.functional as F

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('summerization_loss')
class SummerizationLoss(FairseqCriterion):
    """Implementation for loss of Bert
        Combine masked language model loss as well as sentence-level classfication
        loss
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        #lm_targets = sample['net_input']['src_tokens'].view(-1)
        lm_targets = sample['target'].contiguous().view(-1)
        # mlm loss
        lm_logits = net_output.float()
        nsentences = lm_logits.size(0)
        lm_logits = lm_logits.contiguous().view(-1, lm_logits.size(-1))
        lm_loss = F.cross_entropy(
            lm_logits,
            lm_targets,
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce
        )

        ntokens = utils.strip_pad(lm_targets, self.padding_idx).numel()

        sample_size = nsentences if self.args.sentence_avg else ntokens
        loss = lm_loss / ntokens
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'lm_loss': utils.item(lm_loss.data) if reduce else lm_loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        lm_loss_sum = sum(log.get('lm_loss', 0) for log in logging_outputs)
        sentence_loss_sum = sum(log.get('sentence_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_loss = (
            lm_loss_sum / ntokens / math.log(2)
            +
            sentence_loss_sum / nsentences / math.log(2)
        )

        agg_output = {
            'loss': agg_loss,
            'lm_loss': lm_loss_sum / ntokens / math.log(2),
            'sentence_loss': sentence_loss_sum / nsentences / math.log(2),
            'nll_loss': lm_loss_sum / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
