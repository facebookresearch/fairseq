# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import utils
from . import register_criterion
from fairseq.criterions.adaptive_loss import AdaptiveLoss


@register_criterion('label_smoothed_adaptive_loss')
class LabelSmoothedAdaptiveLoss(AdaptiveLoss):

    """This enables label smoothing added on top of AdaptiveLoss Criterion."""

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            '--label-smoothing', default=0., type=float, metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing'
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample with label smoothing."""
        net_output = model(**sample['net_input'])
        cross_entropy = self.compute_cross_entropy_loss(
            model, net_output, sample, reduce
        )
        smooth_loss = self.compute_smooth_loss(model, net_output, sample, reduce=reduce)
        loss = (1. - self.eps) * cross_entropy + smooth_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'cross_entropy': utils.item(cross_entropy.data) if reduce else cross_entropy.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_smooth_loss(self, model, net_output, sample, reduce=True):
        """Helper function to compute the smoothed loss."""
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        lprobs = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        eps_i = self.eps / lprobs.size(-1)
        if reduce:
            smooth_loss = smooth_loss.sum()
        return smooth_loss * eps_i

