import torch
import torch.nn as nn

from fairseq.modules.quant_noise import quant_noise


class BitLinear(nn.Linear):
    def __init__(self, *args, transpose=False, init_weight=None, **kwargs) -> None:
        super(BitLinear, self).__init__(*args, **kwargs)
        self.transpose = transpose
        if init_weight is not None:
            self.weight = init_weight

    @staticmethod
    def _binarize(x):
        # Scaling factor to minimize l2 error with full-precision weights
        alpha = x.norm(p=1).div(x.nelement())
        x_bin = torch.sign(x).mul(alpha)

        # Disregard binarization in backward for STE
        x_bin = (x_bin - x).detach() + x
        return x_bin

    def forward(self, input):
        # Center weights to zero-mean before binarization
        weight_bin = self._binarize(self.weight - self.weight.mean())
        if self.transpose:
            weight_bin = weight_bin.t()
        return nn.functional.linear(input, weight_bin, self.bias)


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
