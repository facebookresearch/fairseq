# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from torch.nn.modules.loss import _Loss


class FairseqCriterion(_Loss):

    def __init__(self):
        super().__init__()

    def grad_denom(self, samples):
        """Gradient normalization term for DataParallel training."""
        raise NotImplementedError

    def prepare(self, model, sample):
        """Apply criterion-specific modifications to the sample."""
        return sample

    def forward(self, net_output, sample):
        """Compute the loss for the given sample and network output."""
        raise NotImplementedError

    def aggregate(self, losses):
        """Aggregate losses from DataParallel training.

        Takes a list of losses as input (as returned by forward) and
        aggregates them into the total loss for the mini-batch.
        """
        raise NotImplementedError
