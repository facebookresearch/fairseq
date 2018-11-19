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


@register_criterion('squad')
class SquadCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):

        targets = sample['target']
        net_input = sample['net_input']
        paragraph_mask = net_input['paragraph_mask']

        outs = model(**net_input)

        outs = [F.log_softmax(o, dim=-1).view(-1, o.size(-1)) if len(o) > 0 else o for o in outs]
        sample_sizes = (sample['nsentences'], sample['possible_sentences'], sample['possible_sentences'])

        if reduce:
            losses = outs[0].new_zeros(1)
            losses = (losses,) * len(outs)
        else:
            losses = tuple(outs[0].new_zeros(t.numel()) for t in targets)

        for t, o, loss, ss in zip(targets, outs, losses, sample_sizes):
            if len(o) == 0:
                continue
            l = F.nll_loss(o, t.view(-1), size_average=False, ignore_index=self.padding_idx, reduce=reduce)
            if reduce:
                l /= ss
            loss += l

        if reduce:
            reduced_loss = loss = losses[0]
        else:
            loss = losses
            reduced_loss = sum(l.sum() / (ss or 1.0) for l, ss in zip(losses, sample_sizes))

        sample_size = sample['nsentences']

        logging_output = {
            'loss': utils.item(reduced_loss),
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output, outs

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
