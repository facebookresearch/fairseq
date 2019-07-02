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
from . import FairseqCriterion, register_criterion


def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    assert logits.size(0) == targets.size(-1), \
        "Logits and Targets tensor shapes don't match up"

    loss = F.nll_loss(
        F.log_softmax(logits, -1, dtype=torch.float32),
        targets,
        reduction="sum",
        ignore_index=ignore_index,
    )
    return loss


@register_criterion('masked_lm_loss')
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    This optionally also computes the next sentence prediction (NSP) loss and
    adds it to the overall loss based on the specified args. There are three
    cases to consider:
        1) Generic MLM training without NSP loss. In this case sentence_targets
           and sentence_logits are both None.
        2) BERT training without NSP loss. In this case sentence_targets is
           not None but sentence_logits is None and we should not be computing
           a sentence level loss.
        3) BERT training with NSP loss. In this case both sentence_targets and
           sentence_logits are not None and we should be computing a sentence
           level loss. The weight of the sentence level loss is specified as
           an argument.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    @staticmethod
    def add_args(parser):
        """Args for MaskedLM Loss"""
        # Default for masked_lm_only is False so as to not break BERT training
        parser.add_argument('--masked-lm-only', default=False,
                            action='store_true', help='compute MLM loss only')
        parser.add_argument('--nsp-loss-weight', default=1.0, type=float,
                            help='weight for next sentence prediction'
                                 ' loss (default 1)')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        lm_logits, output_metadata = model(**sample["net_input"])

        # reshape lm_logits from (N,T,C) to (N*T,C)
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        lm_targets = sample['lm_target'].view(-1)
        lm_loss = compute_cross_entropy_loss(
            lm_logits, lm_targets, self.padding_idx)

        # compute the number of tokens for which loss is computed. This is used
        # to normalize the loss
        ntokens = utils.strip_pad(lm_targets, self.padding_idx).numel()
        loss = lm_loss / ntokens
        nsentences = sample['nsentences']
        # nsentences = 0

        # Compute sentence loss if masked_lm_only is False
        sentence_loss = None
        if not self.args.masked_lm_only:
            sentence_logits = output_metadata['sentence_logits']
            sentence_targets = sample['sentence_target'].view(-1)
            # This needs to be recomputed due to some differences between
            # TokenBlock and BlockPair dataset. This can be resolved with a
            # refactor of BERTModel which we will do in the future.
            # TODO: Remove this after refactor of BERTModel
            nsentences = sentence_targets.size(0)

            # Check for logits being none which can happen when remove_heads
            # is set to true in the BERT model. Ideally we should set
            # masked_lm_only to true in this case, but that requires some
            # refactor in the BERT model.
            if sentence_logits is not None:
                sentence_loss = compute_cross_entropy_loss(
                    sentence_logits, sentence_targets)

                loss += self.args.nsp_loss_weight * (sentence_loss / nsentences)

        # NOTE: as we are summing up per token mlm loss and per sentence nsp loss
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'lm_loss': utils.item(lm_loss.data) if reduce else lm_loss.data,
            # sentence loss is not always computed
            'sentence_loss': (
                (
                    utils.item(sentence_loss.data) if reduce
                    else sentence_loss.data
                ) if sentence_loss is not None else 0.0
            ),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        lm_loss_sum = sum(log.get('lm_loss', 0) for log in logging_outputs)
        sentence_loss_sum = sum(
            log.get('sentence_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_loss = sum(log.get('loss', 0) for log in logging_outputs)

        agg_output = {
            'loss': agg_loss / sample_size / math.log(2) if sample_size > 0 else 0.,
            'lm_loss': lm_loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.,
            'sentence_loss': sentence_loss_sum / nsentences / math.log(2) if nsentences > 0 else 0.,
            'nll_loss': lm_loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
