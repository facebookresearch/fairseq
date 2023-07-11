# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class LayerSelect(nn.Module):
    """Compute samples (from a Gumbel-Sigmoid distribution) which is used as
    either (soft) weighting or (hard) selection of residual connection.
    https://arxiv.org/abs/2009.13102
    """
    def __init__(self, num_layers, num_logits, soft_select=False, sampling_tau=5.):
        super(LayerSelect, self).__init__()
        self.layer_logits = torch.nn.Parameter(
            torch.Tensor(num_logits, num_layers),
            requires_grad=True,
        )
        self.hard_select = not soft_select
        self.tau = sampling_tau
        self.detach_grad = False
        self.layer_samples = [None] * num_logits

    def sample(self, logit_idx):
        """To leverage the efficiency of distributed training, samples for all
        layers are computed at once for each logit_idx. Logits are parameters
        learnt independent of each other.

        Args:
            logit_idx: The index of logit parameters used for sampling.
        """
        assert logit_idx is not None
        self.samples = self._gumbel_sigmoid(
            self.layer_logits[logit_idx, :].detach()
            if self.detach_grad
            else self.layer_logits[logit_idx, :],
            dim=-1,
            tau=self.tau,
            hard=self.hard_select,
        )
        self.layer_samples[logit_idx] = self.samples

    def forward(self, i):
        sample = self.samples[i]
        return sample

    def _gumbel_sigmoid(
        self, logits, tau=1, hard=False, eps=1e-10, dim=-1, threshold=0.5
    ):
        # ~Gumbel(0,1)
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
        if hard:
            # Straight through.
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).masked_fill(y_soft > threshold, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
