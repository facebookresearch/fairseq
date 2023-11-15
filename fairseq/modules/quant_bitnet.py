import torch
import torch.nn as nn

from fairseq.modules.quant_noise import quant_noise


class BinarizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(input):
        # Scaling factor to minimize l2 error with full-precision weights
        alpha = input.norm(p=1).div(input.nelement())

        # Center weights to zero-mean before binarization
        ctr_input = input - input.mean()
        output = torch.sign(ctr_input).mul(alpha)

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() if ctx.needs_input_grad[0] else None
        return grad_input


class BitLinear(nn.Linear):
    def __init__(self, *args, transpose=False, init_weight=None, **kwargs) -> None:
        super(BitLinear, self).__init__(*args, **kwargs)
        self.transpose = transpose
        if init_weight is not None:
            self.weight = init_weight

    def forward(self, input):
        weight_bin = BinarizerFunction.apply(self.weight)
        bias_bin = (
            BinarizerFunction.apply(self.bias) if self.bias is not None else self.bias
        )
        if self.transpose:
            weight_bin = weight_bin.t()
        return nn.functional.linear(input, weight_bin, bias_bin)


class QuantizeBitLinearMixin:
    def _maybe_build_quantize_linear(
        self, input_dim, output_dim, bias=True, transpose=False, init_weight=None
    ):
        if self.weight_bits == 32:
            layer = quant_noise(
                nn.Linear(input_dim, output_dim, bias=bias),
                self.q_noise,
                self.qn_block_size,
            )
        else:
            layer = BitLinear(
                input_dim,
                output_dim,
                bias=bias,
                transpose=transpose,
                init_weight=init_weight,
            )
        return layer
