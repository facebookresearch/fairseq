# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch.nn.modules.loss import _Loss


class HeadSelectionLoss(_Loss):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kl_weight = getattr(args, "kl_weight", 0.0)

    def forward(self, head_samples, sample_sizes, prior=0.5, eps=1e-7):
        """
        head_scores: (num_tasks, num_layers, num_heads)
        sample_sizes: (num_tasks, )
        """
        kl_loss = (head_samples * (torch.log(head_samples + eps) - math.log(prior))).sum(-1).sum(-1)
        kl_loss /= (torch.numel(head_samples) / head_samples.size(0))
        kl_loss = self.kl_weight * torch.matmul(kl_loss, sample_sizes)
        return kl_loss
