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


@register_criterion('ol_bert_loss')
class DocBertLoss(FairseqCriterion):
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
        target = sample['target']
        sent_pos_targets = target[:, :, 0].contiguous().view(-1)
        mlm_targets = target[:, :, 1:].contiguous().view(-1)
        
        sent_pos_logits, mlm_logits = model(**sample['net_input'], return_predictions=True)
        
        # mlm loss
        mlm_logits = mlm_logits.float().view(-1, mlm_logits.size(-1))
        mlm_loss = F.cross_entropy(
            mlm_logits,
            mlm_targets,
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce
        )
        
        # sentence loss
        sent_pos_logits = sent_pos_logits.float().view(-1, sent_pos_logits.size(-1))
        sent_pos_loss = F.cross_entropy(
            sent_pos_logits,
            sent_pos_targets,
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce
        )
        
        ntokens = utils.item(mlm_targets.ne(self.padding_idx).sum())
        nsentences = utils.item(sent_pos_targets.ne(self.padding_idx).sum())
        
        sample_size = ntokens + nsentences
        loss = mlm_loss / ntokens + sent_pos_loss / nsentences
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'mlm_loss': utils.item(mlm_loss.data) if reduce else lm_loss.data,
            'sent_pos_loss': utils.item(sent_pos_loss.data) if reduce else sent_pos_loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        mlm_loss_sum = sum(log.get('mlm_loss', 0) for log in logging_outputs)
        sent_pos_loss_sum = sum(log.get('sent_pos_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        mlm_loss = mlm_loss_sum / ntokens / math.log(2)
        sent_pos_loss = sent_pos_loss_sum / nsentences / math.log(2)
        agg_loss = mlm_loss + sent_pos_loss

        agg_output = {
            'loss': agg_loss,
            'mlm_loss': mlm_loss,
            'sent_pos_loss': sent_pos_loss,
            'nll_loss': mlm_loss,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
