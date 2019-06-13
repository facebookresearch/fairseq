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


@register_criterion('adaptive_loss')
class AdaptiveLoss(FairseqCriterion):
    """This is an implementation of the loss function accompanying the adaptive softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax approximation for GPUs"
    (http://arxiv.org/abs/1609.04309)."""

    def __init__(self, args, task):
        super().__init__(args, task)

        if args.ddp_backend == 'c10d':
            raise Exception(
                'AdaptiveLoss is not compatible with the c10d '
                'version of DistributedDataParallel. Please use '
                '`--ddp-backend=no_c10d` instead.'
            )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        assert hasattr(model.decoder, 'adaptive_softmax') and model.decoder.adaptive_softmax is not None
        adaptive_softmax = model.decoder.adaptive_softmax

        net_output = model(**sample['net_input'])
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)

        loss = net_output[0].new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert (target[i].min() >= 0 and target[i].max() <= logits[i].size(1))
                loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction='sum' if reduce else 'none',
                )

        orig = utils.strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'nll_loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
