# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..ops import emulate_int


class ActivationQuantizer:
    """
    Fake scalar quantization of the activations using a forward hook.

    Args:
        - module. a nn.Module for which we quantize the *post-activations*
        - p: proportion of activations to quantize, set by default to 1
        - update_step: to recompute quantization parameters
        - bits: number of bits for quantization
        - method: choose among {"tensor", "histogram", "channel"}
        - clamp_threshold: to prevent gradients overflow

    Remarks:
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - For the list of quantization methods and number of bits, see ops.py
        - To remove the hook from the module, simply call self.handle.remove()
        - At test time, the activations are fully quantized
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick
        - The activations are hard-clamped in [-clamp_threshold, clamp_threshold]
          to prevent overflow during the backward pass
    """
    def __init__(self, module, p=1, update_step=1000, bits=8,
                 method="histogram", clamp_threshold=5):
        self.module = module
        self.p = p
        self.update_step = update_step
        self.counter = 0
        self.bits = bits
        self.method = method
        self.clamp_threshold = clamp_threshold
        self.handle = None
        self.register_hook()

    def register_hook(self):
        # forward hook
        def quantize_hook(module, x, y):

            # update parameters every 1000 iterations
            if self.counter % self.update_step == 0:
                self.scale = None
                self.zero_point = None
            self.counter += 1

            # train with QuantNoise and evaluate the fully quantized network
            p = self.p if self.module.training else 1

            # quantize activations
            y_q, self.scale, self.zero_point = emulate_int(
                y.detach(),
                bits=self.bits,
                method=self.method,
                scale=self.scale,
                zero_point=self.zero_point,
            )

            # mask to apply noise
            mask = torch.zeros_like(y)
            mask.bernoulli_(1 - p)
            noise = (y_q - y).masked_fill(mask.bool(), 0)

            # using straight-through estimator (STE)
            clamp_low = - self.scale * self.zero_point
            clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
            return torch.clamp(y, clamp_low.item(), clamp_high.item()) + noise.detach()

        # register hook
        self.handle = self.module.register_forward_hook(quantize_hook)
