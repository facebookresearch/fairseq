# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion("label_smoothed_cross_entropy_with_accuracy")
class LabelSmoothedCrossEntropyWithAccuracyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss with accuracy metric for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, correct, total = self.compute_loss_and_acc(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data),
            'nll_loss': utils.item(nll_loss.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'correct': utils.item(correct.data),
            'total': utils.item(total.data),
        }
        return loss, sample_size, logging_output

    def compute_loss_and_acc(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if not lprobs.batch_first:
            lprobs = lprobs.transpose(0, 1)
        lprobs = lprobs.view(-1, lprobs.size(-1))  # -> (B x T) x C
        target = model.get_targets(sample, net_output).view(-1)  # -> (B x T)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        mask = target.ne(self.padding_idx)
        correct = torch.sum(lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)

        return loss, nll_loss, correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)

        correct_sum = utils.item(sum(log.get('correct', 0) for log in logging_outputs))
        metrics.log_scalar('correct', correct_sum)
        total_sum = utils.item(sum(log.get('total', 0) for log in logging_outputs))
        metrics.log_scalar('total', total_sum)
        if total_sum > 0:
            metrics.log_derived(
                'acc',
                lambda meters: round(
                    meters['correct'].sum * 100.0 / meters['total'].sum, 3
                ) if meters['total'].sum > 0 else float('nan'),
            )

        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
