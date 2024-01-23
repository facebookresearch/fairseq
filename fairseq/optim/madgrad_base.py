# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.optim

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from fairseq.optim import quant_utils

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class MADGRAD(torch.optim.Optimizer):
    """
    MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimization.

    .. _MADGRAD: https://arxiv.org/abs/2101.11075

    MADGRAD is a general purpose optimizer that can be used in place of SGD or
    Adam may converge faster and generalize better. Currently GPU-only.
    Typically, the same learning rate schedule that is used for SGD or Adam may
    be used. The overall learning rate is not comparable to either method and
    should be determined by a hyper-parameter sweep.

    MADGRAD requires less weight decay than other methods, often as little as
    zero. Momentum values used for SGD or Adam's beta1 should work here also.

    On sparse problems both weight_decay and momentum should be set to 0.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation to improve
            numerical stability. (default: 1e-6).
            This parameter is less important in MADGRAD than in Adam.
            On problems with very small gradients, setting this to 0 will improve convergence.
        decouple_decay (bool):
            Apply AdamW style decoupled weight decay (EXPERIMENTAL).

    """

    def __init__(
        self,
        params: _params_t,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
        decouple_decay=False,
        quant_method="ste",
        quant_bits=32,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1)")
        if lr < 0:
            raise ValueError(f"Learning rate {lr} must be non-negative")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps {eps} must be non-negative")

        defaults = dict(
            lr=lr,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            decouple_decay=decouple_decay,
            quant_method=quant_method,
            quant_bits=quant_bits,
        )
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return True

    @staticmethod
    def _compute_z_rms(
        state, p_data_fp32, grad, momentum, eps, lamb, inplace=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = state["s"]
        grad_sum_sq = state["grad_sum_sq"]
        if momentum == 0:
            # Compute x_0 from other known quantities
            rms = grad_sum_sq.pow(1 / 3).add_(eps)
            x0 = p_data_fp32.addcdiv(s, rms, value=1)
        else:
            x0 = state["x0"]

        # Accumulate second moments
        if inplace:
            grad_sum_sq.addcmul_(grad, grad, value=lamb)
        else:
            grad_sum_sq = grad_sum_sq.addcmul(grad, grad, value=lamb)

        rms = grad_sum_sq.pow(1 / 3).add_(eps)

        if eps == 0:
            rms[rms == 0] = float("inf")

        # Update s
        if inplace:
            s.data.add_(grad, alpha=lamb)
        else:
            s = s.add(grad, alpha=lamb)

        # Step
        z = x0.addcdiv(s, rms, value=-1)
        return z, rms

    @staticmethod
    def init_p_and_grad(state, p, momentum):
        grad = p.grad.data
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()

        p_data_fp32 = p.data
        if p.data.dtype in {torch.float16, torch.bfloat16}:
            p_data_fp32 = p_data_fp32.float()

        if "grad_sum_sq" not in state:
            state["grad_sum_sq"] = torch.zeros_like(p_data_fp32)
            state["s"] = torch.zeros_like(p_data_fp32)

        if momentum != 0 and "x0" not in state:
            state["x0"] = p_data_fp32.clone()

        return p_data_fp32, grad

    @staticmethod
    def binarize_param(latent_p, p_data_fp32):
        omega = quant_utils.estimate_omega(latent_p)
        quant_utils.scaled_sign_(p_data_fp32.copy_(latent_p), omega)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if "k" not in self.state:
            self.state["k"] = torch.tensor([0], dtype=torch.long)
        k = self.state["k"].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"]
            if lr != 0.0:
                lr = lr + eps  # For stability
            decay = group["weight_decay"]
            momentum = group["momentum"]
            decouple_decay = group["decouple_decay"]
            quant_bits = group["quant_bits"]
            quant_method = group["quant_method"]

            apply_par = quant_bits == 1 and quant_method == "parq"
            apply_ste = quant_bits == 1 and quant_method == "ste"

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                p_data_fp32, grad = self.init_p_and_grad(state, p, momentum)

                if apply_par and "lamb_sum" not in state:
                    state["lamb_sum"] = torch.zeros(
                        1, dtype=p_data_fp32.dtype, device=p_data_fp32.device
                    )
                    state["z"] = torch.zeros_like(p_data_fp32)
                if apply_ste and "latent_p" not in state:
                    state["latent_p"] = p_data_fp32.clone()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError(
                        "momentum != 0 is not compatible with sparse gradients"
                    )

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                # Apply weight decay
                if decay != 0 and not decouple_decay:
                    if grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )

                    grad.add_(p_data_fp32, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p_data_fp32.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    x0_masked_vals = p_masked._values().addcdiv(
                        s_masked._values(), rms_masked_vals, value=1
                    )

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)

                    if eps == 0:
                        rms_masked_vals[rms_masked_vals == 0] = float("inf")

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(
                        s_masked._values(), rms_masked_vals, value=-1
                    )
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    z, rms = self._compute_z_rms(
                        state, p_data_fp32, grad, momentum, eps, lamb, inplace=True
                    )

                    if apply_par:
                        rms.div_(state["lamb_sum"].add_(lamb)).clamp_(max=1)
                        omega = quant_utils.estimate_omega(z)
                        z = torch.where(
                            z.abs() < omega.mul(rms),
                            z.div(rms),
                            torch.sign(z).mul_(omega),
                        )
                        state["z"].copy_(z)

                    if momentum != 0:
                        z.mul_(ck).add_(
                            state["latent_p"] if apply_ste else p_data_fp32,
                            alpha=1 - ck,
                        )

                    if apply_ste:
                        state["latent_p"].copy_(z)
                        omega = quant_utils.estimate_omega(z)
                        quant_utils.scaled_sign_(z, omega)

                    if decouple_decay and decay != 0:
                        p_old = p_data_fp32.clone()

                    p_data_fp32.copy_(z)

                    if decouple_decay and decay != 0:
                        p_data_fp32.add_(p_old, alpha=-lr * decay)

                    if p.data.dtype in {torch.float16, torch.bfloat16}:
                        p.data.copy_(p_data_fp32)

        self.state["k"] += 1
        return loss
