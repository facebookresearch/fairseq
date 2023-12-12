import math
import torch
import torch.distributed as dist

from functools import partial
from torch import nn


class QParams(nn.Module):
    def __init__(self, n_bits, device, requires_grad: bool = True):
        super().__init__()
        q_base = 2 ** (n_bits - 1)
        self.q_min = -q_base
        self.q_max = q_base - 1

        factory_kwargs = {"device": device, "requires_grad": requires_grad}

        self.scale = nn.Parameter(torch.ones(1, **factory_kwargs))
        self.zero_point = nn.Parameter(torch.zeros(1, **factory_kwargs))
        self.register_buffer("eps", torch.tensor([torch.finfo(torch.float32).eps]))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    @staticmethod
    def reduce_tensor(X):
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if world_size > 1 and X.is_cuda:
            dist.all_reduce(X / world_size)

    def is_initialized(self):
        return self.initialized.item()

    @torch.no_grad()
    def _maybe_initialize(self, data):
        if not self.is_initialized():
            tensor_norm = data.abs().mean()
            self.reduce_tensor(tensor_norm)
            self.scale.copy_(2 * tensor_norm / math.sqrt(self.q_max))
            self.scale.data.clamp_(min=self.eps.item())

            self.initialized = torch.ones_like(self.initialized)

    def forward(self, X):
        self._maybe_initialize(X)
        grad_factor = 1.0 / math.sqrt(X.numel() * self.q_max)
        output = torch._fake_quantize_learnable_per_tensor_affine(
            X, self.scale, self.zero_point, self.q_min, self.q_max, grad_factor
        )
        return output


def quantizer_function(input: torch.tensor, qparams: QParams):
    """Based on Learned Step Size Quantization: https://arxiv.org/abs/1902.08153

    Cannot inherit from torch.autograd.Function since only the forward pass exists.
    """
    output = qparams(input)
    return output


def get_default_quant_fn(weight, weight_bits):
    if weight_bits > 1:
        qparams = QParams(weight_bits, weight.device)
        quant_fn = quantizer_function
    else:
        qparams = None
        quant_fn = BinarizerFunction.apply
    return qparams, quant_fn


class BinarizerFunction(torch.autograd.Function):
    generate_vmap_rule = True

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
    def __init__(
        self,
        *args,
        weight_bits=1,
        transpose=False,
        init_weight=None,
        binarize_per_ch=False,
        **kwargs
    ) -> None:
        super(BitLinear, self).__init__(*args, **kwargs)
        if weight_bits > 1 and binarize_per_ch:
            raise NotImplementedError

        self.transpose = transpose
        self.qparams, self.quant_fn = get_default_quant_fn(self.weight, weight_bits)
        self.weight_bin_fn = partial(
            (self._channel_binarizer if binarize_per_ch else self._default_binarizer),
            quant_fn=self.quant_fn,
            n_bits=weight_bits,
        )

        if init_weight is not None:
            self.weight = init_weight

    @staticmethod
    def _default_binarizer(x, qparams, quant_fn=BinarizerFunction, n_bits=1):
        return quant_fn(x, qparams) if n_bits > 1 else quant_fn(x)

    @staticmethod
    def _channel_binarizer(x, qparams, quant_fn=BinarizerFunction, n_bits=1):
        vm = torch.vmap(quant_fn)
        return vm(x, qparams, n_bits) if n_bits > 1 else vm(x)

    def quantize_weights(self):
        return self.weight_bin_fn(self.weight, self.qparams)

    def forward(self, input):
        weight_bin = self.quantize_weights()
        if self.transpose:
            weight_bin = weight_bin.t()
        return nn.functional.linear(input, weight_bin, self.bias)


class QuantizeBitLinearMixin:
    def _maybe_build_quantize_linear(
        self,
        input_dim,
        output_dim,
        bias=True,
        transpose=False,
        init_weight=None,
        binarize_per_ch=False,
    ):
        if self.weight_bits < 32:
            layer = BitLinear(
                input_dim,
                output_dim,
                bias=bias,
                transpose=transpose,
                init_weight=init_weight,
                binarize_per_ch=binarize_per_ch,
                weight_bits=self.weight_bits,
            )
        else:
            layer = nn.Linear(input_dim, output_dim, bias=bias)
        return layer
