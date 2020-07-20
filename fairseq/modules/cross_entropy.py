# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def _cross_entropy_pytorch(logits, target, ignore_index=None, reduction='mean'):
    lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    return F.nll_loss(
        lprobs, target, ignore_index=ignore_index, reduction=reduction,
    )


try:
    import xentropy_cuda
    from apex.contrib import xentropy

    logger.info('using fused cross entropy')

    def cross_entropy(logits, target, ignore_index=-100, reduction='mean'):
        if logits.device == torch.device('cpu'):
            return _cross_entropy_pytorch(logits, target, ignore_index, reduction)
        else:
            half_to_float = (logits.dtype == torch.half)
            losses = xentropy.SoftmaxCrossEntropyLoss.apply(
                logits, target, 0.0, ignore_index, half_to_float,
            )
            if reduction == 'sum':
                return losses.sum()
            elif reduction == 'mean':
                if ignore_index >= 0:
                    return losses.sum() / target.ne(ignore_index).sum()
                else:
                    return losses.mean()
            elif reduction == 'none':
                return losses
            else:
                raise NotImplementedError

except ImportError:

    def cross_entropy(logits, target, ignore_index=-100, reduction='mean'):
        return _cross_entropy_pytorch(logits, target, ignore_index, reduction)
