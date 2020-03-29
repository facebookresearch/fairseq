#!/usr/bin/env python3

class ActivationQuantizer:
    """
    Fake scalar quantization of the activations using a forward hook.

    Args:
        - module. a nn.Module for which we quantize the *post-activations*
        - p: proportion of activations to quantize, set by default to 1
        - bits: number of bits for quantization
        - method: choose among {"tensor", "histogram", "channel"}

    Remarks:
        - Parameters scale and zero_point are calculated only once,
          during the first call of the hook
        - Different quantization methods and number of bits, see ops.py
        - To remove the hook from the module, simply call self.handle.remove()
        - At test time, the activations are fully quantized
    """
    def __init__(self, module, p=1, bits=8, method="tensor"):
        self.module = module
        self.p = p
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
                y.detach(),
                bits=self.bits,
                method=self.method,
                scale=self.scale,
                zero_point=self.zero_point,
            )
            
            # mask to apply noise
            mask = torch.zeros_like(y)
            mask.bernoulli_(1 - self.p)
            noise = (y_q - y).masked_fill(mask.bool(), 0)

            # using straight-through estimator (STE)
            return y + noise.detach()

        # register hook
        self.handle = self.module.register_forward_hook(quantize_hook)
