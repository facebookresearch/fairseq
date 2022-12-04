# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_val):
        ctx.clip_val = clip_val
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        cg = torch.clip(grad, min=ctx.clip_val, max=ctx.clip_val), None
        return cg
