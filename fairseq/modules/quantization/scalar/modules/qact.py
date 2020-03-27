#!/usr/bin/env python3
from ..ops import emulate_int


class ActivationQuantizer:
    """
    Fake scalar quantization of the activations using a forward hook.

    Args:
        - module. a nn.Module for which we quantize the *post-activations*
        - bits: number of bits for quantization
        - method: choose among {"tensor", "histogram", "channel"}

    Remarks:
        - Parameters scale and zero_point are calculated only once,
          during the first call of the hook
        - Different quantization methods and number of bits, see ops.py
        - To remove the hook from the module, simply call self.handle.remove()
    """
    def __init__(self, module, bits=8, method="histogram"):
        self.module = module
        self.bits = bits
        self.method = method
        self.scale = None
        self.zero_point = None
        self.handle = None
        self.register_hook()

    def register_hook(self):
        # forward hook
        def quantize_hook(module, x, y):
            y_q, self.scale, self.zero_point = emulate_int(
                y,
                bits=self.bits,
                method=self.method,
                scale=self.scale,
                zero_point=self.zero_point,
            )
            return y_q

        # register hook
        self.handle = self.module.register_forward_hook(quantize_hook)
