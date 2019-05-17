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
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('odd_one_out')
class OddOneOutCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        assert self.args.task == 'odd_one_out_lm'

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--ooo-weight', default=1., type=float, metavar='W',
                            help='weight for Odd-One-Out loss term')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert not self.args.sentence_avg

        # compute cloze loss
        features, _ = model.extract_features(**sample['net_input'])
        logits = model.output_layer(features)
        targets = model.get_targets(sample, [logits])
        loss = F.nll_loss(
            F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32),
            targets.view(-1),
            ignore_index=self.padding_idx,
            reduction='sum',
        )
        cloze_loss = utils.item(loss)
        loss /= sample['ntokens']

        # compute odd-one-out loss
        ooo_sample_size = sample['ooo_endpoint_labels'].numel()
        if ooo_sample_size > 0:
            # extract sentence endpoints
            num_features = features.size(-1)
            x = features.contiguous().view(-1, num_features)
            x = x[sample['ooo_endpoints']]
            x = x.view(-1, 2, num_features)
            ooo_logits = model.ooo_head(x, padding_mask=None)

            # add odd-one-out loss to existing cloze loss
            ooo_loss = F.nll_loss(
                F.log_softmax(ooo_logits, dim=-1, dtype=torch.float32),
                sample['ooo_endpoint_labels'],
                reduction='sum',
            )
            loss += self.args.ooo_weight * (ooo_loss / ooo_sample_size)
        else:
            ooo_loss = 0.

        sample_size = 1
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'cloze_loss': cloze_loss,
            'ooo_loss': utils.item(ooo_loss),
            'ooo_label_sum': utils.item(sample['ooo_endpoint_labels'].sum()),
            'ooo_sample_size': ooo_sample_size,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        cloze_loss_sum = sum(log.get('cloze_loss', 0) for log in logging_outputs)
        ooo_loss_sum = sum(log.get('ooo_loss', 0) for log in logging_outputs)
        ooo_label_sum = sum(log.get('ooo_label_sum', 0) for log in logging_outputs)
        ooo_sample_size = sum(log.get('ooo_sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size,
            'cloze_loss': cloze_loss_sum / ntokens / math.log(2),
            'ooo_loss': ooo_loss_sum / max(ooo_sample_size, 1),
            'ooo_baseline': ooo_label_sum / max(ooo_sample_size, 1),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
