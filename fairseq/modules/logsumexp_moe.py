# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch


class LogSumExpMoE(torch.autograd.Function):
    """Standard LogSumExp forward pass, but use *posterior* for the backward.

    See `"Mixture Models for Diverse Machine Translation: Tricks of the Trade"
    (Shen et al., 2019) <https://arxiv.org/abs/1902.07816>`_.
    """

    @staticmethod
    def forward(ctx, logp, posterior, dim=-1):
        ctx.save_for_backward(posterior)
        ctx.dim = dim
        return torch.logsumexp(logp, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        posterior, = ctx.saved_tensors
        grad_logp = grad_output.unsqueeze(ctx.dim) * posterior
        return grad_logp, None, None
