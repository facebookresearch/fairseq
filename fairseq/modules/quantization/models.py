import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from .quant import Binarizer, get_qminmax, get_scale_init, QuantLS2, LSQClampRound


class AdaptedLinear(nn.Linear):
    def __init__(self, weight_init: Tensor, bias_init: Optional[Tensor]):
        factory_kwargs = {"dtype": weight_init.dtype, "device": weight_init.device}
        out_features, in_features = weight_init.shape
        bias = bias_init is not None
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)
        self._weight = nn.Parameter(self.weight.detach().clone())
        delattr(self, "weight")

        if self.bias is not None:
            self.bias.data.copy_(bias_init)


class BinarizedLinear(AdaptedLinear):
    @property
    def weight(self):
        return Binarizer.apply(self._weight)


class LSQLinear(AdaptedLinear):
    def __init__(
        self, weight_init: Tensor, bias_init: Optional[Tensor], quant_bits: int
    ):
        super().__init__(weight_init, bias_init)

        self.qmin, self.qmax = get_qminmax(quant_bits)
        self.scale = nn.Parameter(get_scale_init(self._weight, self.qmax))

    @property
    def weight(self):
        return LSQClampRound.apply(self._weight, self.scale, self.qmin, self.qmax)

    def extra_repr(self):
        qmin, qmax = self.qmin, self.qmax
        return ", ".join([super().extra_repr(), f"{qmin=}", f"{qmax=}"])


class QuantLS2Linear(AdaptedLinear):
    def __init__(
        self, weight_init: Tensor, bias_init: Optional[Tensor], stride: int = 5
    ):
        super().__init__(weight_init, bias_init)

        self.register_buffer("v1", torch.zeros(1))
        self.register_buffer("v2", torch.zeros(1))
        self.stride = stride

    @property
    def weight(self):
        return QuantLS2.apply(
            self._weight, self.v1, self.v2, self.stride, self.training
        )

    def extra_repr(self):
        stride = self.stride
        return ", ".join([super().extra_repr(), f"{stride=}"])


def get_quant_module(
    quant_bits: int, quant_method: str, weight_init: Tensor, bias_init: Tensor
) -> AdaptedLinear:
    module_cls = None
    kwargs = {}
    if quant_method == "lsq":
        module_cls = LSQLinear
        kwargs["quant_bits"] = quant_bits
    elif quant_method == "least-sq" and quant_bits < 3:
        module_cls = BinarizedLinear if quant_bits == 1 else QuantLS2Linear
    else:
        raise NotImplementedError

    return module_cls(weight_init, bias_init, **kwargs)
