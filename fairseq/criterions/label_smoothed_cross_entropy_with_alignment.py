# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from . import register_criterion


@register_criterion('label_smoothed_cross_entropy_with_alignment')
class LabelSmoothedCrossEntropyCriterionWithAlignment(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.alignment_lambda = args.alignment_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        super(LabelSmoothedCrossEntropyCriterionWithAlignment,
              LabelSmoothedCrossEntropyCriterionWithAlignment).add_args(parser)
        parser.add_argument('--alignment-lambda', default=0.05, type=float, metavar='D',
                            help='weight for the alignment loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        alignment_loss = None

        # Compute alignment loss only for training set and non dummy batches.
        if 'alignments' in sample and sample['alignments'] is not None:
            alignment_loss = self.compute_alignment_loss(sample, net_output)

        if alignment_loss is not None:
            logging_output['alignment_loss'] = utils.item(alignment_loss.data)
            loss += self.alignment_lambda * alignment_loss

        return loss, sample_size, logging_output

    def compute_alignment_loss(self, sample, net_output):
        attn_prob = net_output[1]['attn']
        bsz, tgt_sz, src_sz = attn_prob.shape
        attn = attn_prob.view(bsz * tgt_sz, src_sz)

        align = sample['alignments']
        align_weights = sample['align_weights'].float()

        if len(align) > 0:
            # Alignment loss computation. align (shape [:, 2]) contains the src-tgt index pairs corresponding to
            # the alignments. align_weights (shape [:]) contains the 1 / frequency of a tgt index for normalizing.
            loss = -((attn[align[:, 1][:, None], align[:, 0][:, None]]).log() * align_weights[:, None]).sum()
        else:
            return None

        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'alignment_loss': sum(log.get('alignment_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `aggregate_logging_outputs`.
        Setting this to True will improves distributed training speed.
        """
        return True
