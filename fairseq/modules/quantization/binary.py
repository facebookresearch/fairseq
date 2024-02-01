import torch

from fairseq.modules.quantization.adapted_linear import AdaptedLinear


class Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        omega = input.abs().mean()
        return input.sign().mul_(omega)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output if ctx.needs_input_grad[0] else None


class BinarizedLinear(AdaptedLinear):
    def __init__(self, weight_init, bias_init):
        super().__init__(weight_init, bias_init)

    @property
    def weight(self):
        return Binarizer.apply(self._weight)
