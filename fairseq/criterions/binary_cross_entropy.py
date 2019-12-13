# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('binary_cross_entropy')
class BinaryCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output, expand_steps=False).float()

        if hasattr(model, 'get_target_weights'):
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()
        else:
            weights = 1.

        loss = F.binary_cross_entropy_with_logits(logits, target, reduce=False)

        loss = loss * weights

        if reduce:
            loss = loss.sum()

        sample_size = target.numel()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample_size,
            'nsentences': logits.size(0),
            'sample_size': sample_size,
        }
        if log_pred:
            logging_output['logits'] = logits.cpu().numpy()
            logging_output['target'] = target.cpu().numpy()
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
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        for key in ["logits", "target"]:
            if key in logging_outputs[0]:
                if len(logging_outputs) == 1:
                    agg_output[key] = logging_outputs[0][key]  # avoid copying
                else:
                    agg_output[key] = np.concatenate(
                        [log[key] for log in logging_outputs]
                    )
        return agg_output
