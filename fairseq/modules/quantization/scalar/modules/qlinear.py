# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import emulate_int


class IntLinear(nn.Module):
    """
    Quantized counterpart of the nn.Linear module that applies QuantNoise during training.

    Args:
        - in_features: input features
        - out_features: output features
        - bias: bias or not
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick.
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        p=0,
        update_step=3000,
        bits=8,
        method="histogram",
    ):
        super(IntLinear, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.chosen_bias = bias
        if self.chosen_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # quantization parameters
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.chosen_bias:
            nn.init.constant_(self.bias, 0.0)
        return

    def forward(self, input):
        # train with QuantNoise and evaluate the fully quantized network
        p = self.p if self.training else 1

        # update parameters every 100 iterations
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1

        # quantize weight
        weight_quantized, self.scale, self.zero_point = emulate_int(
            self.weight.detach(),
            bits=self.bits,
            method=self.method,
            scale=self.scale,
            zero_point=self.zero_point,
        )

        # mask to apply noise
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)

        # using straight-through estimator (STE)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2**self.bits - 1 - self.zero_point)
        weight = (
            torch.clamp(self.weight, clamp_low.item(), clamp_high.item())
            + noise.detach()
        )

        # return output
        output = F.linear(input, weight, self.bias)
        return output

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, quant_noise={}, bits={}, method={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.p,
            self.bits,
            self.method,
        )
