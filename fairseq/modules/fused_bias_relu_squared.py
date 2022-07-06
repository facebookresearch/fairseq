# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


@torch.jit.script
def bias_relu_squared(bias, y):
    x = bias + y
    return F.relu(x).pow(2)


@torch.jit.script
def bias_relu_squared_backward(g, bias, y):
    relu_out = F.relu(bias + y)
    relu_sq_deriv = 2 * relu_out
    return relu_sq_deriv * g


class ReluSquaredFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_relu_squared(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_relu_squared_backward(grad_output, bias, input)
        return tmp, tmp


fused_bias_relu_squared = ReluSquaredFunction.apply
