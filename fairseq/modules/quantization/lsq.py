import math
import torch
import torch.nn.functional as F

from torch import nn


class LSQClampRound(torch.autograd.Function):
    @staticmethod
    def forward(target, scale, qmin, qmax):
        quant_target = target.div(scale).clamp_(qmin, qmax)
        return quant_target.round().mul_(scale)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        target, scale, qmin, qmax = inputs
        ctx.save_for_backward(target, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax

    @staticmethod
    def backward(ctx, grad_output):
        target, scale = ctx.saved_tensors
        grad_target = grad_scale = None

        quant_target = target.div(scale)
        neg_mask = quant_target.lt(ctx.qmin).float()
        pos_mask = quant_target.gt(ctx.qmax).float()
        mid_mask = 1.0 - neg_mask - pos_mask

        if ctx.needs_input_grad[0]:
            grad_target = grad_output.mul(mid_mask)

        if ctx.needs_input_grad[1]:
            neg_mask.mul_(ctx.qmin)
            pos_mask.mul_(ctx.qmax)
            grad_scale_factor = 1.0 / math.sqrt(ctx.qmax * target.numel())
            mid_mask.mul_(quant_target.round().sub_(quant_target)).mul_(
                grad_output * grad_scale_factor
            )
            grad_scale = neg_mask + pos_mask + mid_mask

        return grad_target, grad_scale, None, None


class LSQLinear(nn.Linear):
    def __init__(
        self,
        quant_bits: int,
        weight_init: torch.Tensor,
        bias_init: torch.Tensor,
    ):
        factory_kwargs = {"device": weight_init.device, "dtype": weight_init.dtype}
        in_features = weight_init.shape[1]
        out_features = weight_init.shape[0]
        super().__init__(
            in_features, out_features, bias=bias_init is not None, **factory_kwargs
        )
        q_base = 2 ** (quant_bits - 1)
        self.qmin = -q_base
        self.qmax = q_base - 1

        delattr(self, "weight")
        self._weight = nn.Parameter(weight_init)

        self.scale = nn.Parameter(2 * weight_init.abs().mean() / math.sqrt(self.qmax))

    @property
    def weight(self):
        return LSQClampRound.apply(self._weight, self.scale, self.qmin, self.qmax)

    def extra_repr(self):
        qmin, qmax = self.qmin, self.qmax
        return ", ".join([super().extra_repr(), f"{qmin=}", f"{qmax=}"])
