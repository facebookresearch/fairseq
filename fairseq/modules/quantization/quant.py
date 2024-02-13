import math
import torch

from torch.nested import nested_tensor
from typing import Tuple, Type


def get_qminmax(quant_bits) -> Tuple[int, int]:
    q_base = 2 ** (quant_bits - 1)
    qmin = -q_base
    qmax = q_base - 1
    return qmin, qmax


def get_quant_cls(quant_bits) -> Type[torch.autograd.Function]:
    assert quant_bits < 32
    return Binarizer if quant_bits == 1 else LSQClampRound


def get_scale_init(input, qmax):
    return 2 * input.abs().mean() / math.sqrt(qmax)


def l1_normalized(input, dim=None, keepdim=False) -> torch.Tensor:
    return input.abs().mean(dim=dim, keepdim=keepdim)


def scaled_sign(input) -> torch.Tensor:
    omega = l1_normalized(input)
    return input.sign().mul_(omega)


class Binarizer(torch.autograd.Function):
    """1-bit binary quantization: https://arxiv.org/abs/1603.05279"""

    @staticmethod
    def forward(ctx, input):
        return scaled_sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output if ctx.needs_input_grad[0] else None


class QuantLS2(torch.autograd.Function):
    """Least squares 2-bit binary quantization: https://arxiv.org/abs/2001.02786"""

    @staticmethod
    def get_cand_mask(values, values2, cmean):
        """Get mask for candidate values of `v1`.

        Args:
            values: Input at `slice(1, -1)`
            values2: Input at `slice(2, None)`
            cmean: Cumulative mean at `slice(1, -1)`
        """
        return values.le(cmean).logical_and_(values2.ge(cmean))

    @staticmethod
    def collect_masked(values, mask, dim):
        """Given 1D `values` and 2D `mask`, scatter `values` into 2D matrix."""
        split_sizes = mask.sum(dim=dim).tolist()
        masked = values[mask].split(split_sizes)
        masked = nested_tensor(list(masked)).to_padded_tensor(0.0)
        return masked

    @staticmethod
    def get_v1_cands(input, dim: int = -1):
        """Get candidate values of `v1` using Eq. 10.

        Args:
            input: Absolute value of original input.
            dim: Dimension to reduce over.
        """
        values = input.sort(dim=dim).values

        # Cumulative sum over columns in csum; total row-wise sum in csum_r
        csum = values.cumsum(dim=dim)
        csum, csum_r = csum[:, 1:-1], csum[:, -1:]

        # Cumulative mean: left-to-right in cmean_fw, reverse in cmean_bw
        counts = torch.arange(1, input.size(dim=dim), device=input.device)
        cmean_bw = csum_r.sub(csum).div_(counts[:-1].flip(dims=[dim])).div_(2)
        cmean_fw = csum.div(2 * counts[1:]).add_(cmean_bw)

        # Extract mask to estimate conditional expectation
        values, values2 = values[:, 1:-1], values[:, 2:]
        mask = QuantLS2.get_cand_mask(values, values2, cmean_bw).logical_or_(
            QuantLS2.get_cand_mask(values, values2, cmean_fw)
        )
        values = QuantLS2.collect_masked(values, mask, dim)
        return values

    @staticmethod
    def update_residual(residual, v):
        """Update `residual` by its element-wise sign scaled by `v`."""
        return residual.sub(residual.sign().mul(v))

    @staticmethod
    def compute_v1(input, v1s, dim: int = -1):
        """Apply objective function in Eq. 8 over `v1s` and return the best."""
        residual = input.unsqueeze(dim - 1)
        v = v1s.unsqueeze(dim)
        residual = QuantLS2.update_residual(residual, v)
        v = l1_normalized(residual, dim=dim, keepdim=True)
        residual = QuantLS2.update_residual(residual, v)

        costs = residual.norm(dim=dim)
        indices = costs.argmin(dim=-1, keepdim=True)
        v1 = v1s.gather(1, indices).mean()
        return v1

    @staticmethod
    def forward(ctx, input, v1, v2, stride, training: bool = True):
        if training:
            # Simulate striding along randomly permuted columns
            n_col = input.size(1) // stride
            idx = torch.randperm(input.size(1))[:n_col]
            strided_input = input[:, idx].abs()

            v1s = QuantLS2.get_v1_cands(strided_input)
            v1.copy_(QuantLS2.compute_v1(strided_input, v1s))

            residual = QuantLS2.update_residual(input, v1)
            v2.copy_(l1_normalized(residual))

        # Use v1 and v2 to compute quantized input
        input_sgn = input.sign().mul(v1)
        input_sgn.add_((input - input_sgn).sign_().mul_(v2))
        return input_sgn

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output if ctx.needs_input_grad[0] else None
        return grad_input, None, None, None, None


class QuantLSGreedy(torch.autograd.Function):
    """k-bit greedy quantization: https://arxiv.org/abs/1603.05279"""

    @staticmethod
    def forward(ctx, input, vs, training: bool = True):
        residual = input
        input_sgn = torch.zeros_like(input)
        for i in range(vs.size(0)):
            if training:
                vs[i].copy_(l1_normalized(residual))

            residual = QuantLS2.update_residual(residual, vs[i])
            input_sgn.add_((input - input_sgn).sign_().mul_(vs[i]))

        return input_sgn

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output if ctx.needs_input_grad[0] else None
        return grad_input, None, None


class LSQClampRound(torch.autograd.Function):
    """Learned step size quantization: https://arxiv.org/abs/1902.08153"""

    @staticmethod
    def forward(target, scale, qmin, qmax):
        quant_target = target.div(scale).clamp_(qmin, qmax)
        quant_target = quant_target.round().mul_(scale)
        return quant_target

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        target, scale, qmin, qmax = inputs
        ctx.save_for_backward(target, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax

    @staticmethod
    def get_grad_masks(quant_target, qmin, qmax):
        neg_mask = quant_target.lt(qmin).float()
        pos_mask = quant_target.gt(qmax).float()
        mid_mask = 1.0 - neg_mask - pos_mask
        return neg_mask, mid_mask, pos_mask

    @staticmethod
    def get_grad_scale(
        quant_target, qmin, qmax, neg_mask, mid_mask, pos_mask, grad_output
    ):
        mid_mask = mid_mask.mul(quant_target.round().sub(quant_target))
        grad_scale = neg_mask.mul(qmin) + mid_mask + pos_mask.mul(qmax)

        grad_scale_factor = 1.0 / math.sqrt(qmax * quant_target.numel())
        grad_scale.mul_(grad_output * grad_scale_factor)
        return grad_scale

    @staticmethod
    def get_grad_target(grad_output, mid_mask):
        return grad_output.mul(mid_mask)

    @staticmethod
    def backward(ctx, grad_output):
        target, scale = ctx.saved_tensors
        grad_target = grad_scale = None

        quant_target = target.div(scale)
        neg_mask, mid_mask, pos_mask = LSQClampRound.get_grad_masks(
            quant_target, ctx.qmin, ctx.qmax
        )

        if ctx.needs_input_grad[1]:
            grad_scale = LSQClampRound.get_grad_scale(
                quant_target,
                ctx.qmin,
                ctx.qmax,
                neg_mask,
                mid_mask,
                pos_mask,
                grad_output,
            )

        if ctx.needs_input_grad[0]:
            grad_target = LSQClampRound.get_grad_target(grad_output, mid_mask)

        return grad_target, grad_scale, None, None
