import torch

from typing import Any, Dict

from fairseq.optim.quant import (
    get_qminmax,
    get_quant_cls,
    get_scale_init,
)


class QuantOptimizer(torch.optim.Optimizer):
    def add_param_group(self, group: Dict[str, Any]) -> None:
        super().add_param_group(group)

        quant_bits = group["quant_bits"]
        if quant_bits == 32:
            return

        quant_method = group["quant_method"]
        if quant_method == "lsq":
            # Set LSQ constants
            group["qmin"], group["qmax"] = get_qminmax(quant_bits)

        for p in group["params"]:
            old_p = p.clone().detach()  # make full precision copy

            group["quant_cls"] = get_quant_cls(quant_method, quant_bits)
            if quant_method == "lsq":
                # Initialize LSQ scale params
                scale = torch.empty(1, device=old_p.device, dtype=old_p.dtype)
                scale.copy_(get_scale_init(old_p, group["qmax"]))
                p.data.copy_(
                    group["quant_cls"].apply(old_p, scale, group["qmin"], group["qmax"])
                )
                self.state[p]["scale"] = scale
            elif quant_method == "least-sq":
                vs = torch.empty(quant_bits, device=old_p.device, dtype=old_p.dtype)
                p.data.copy_(group["quant_cls"].apply(old_p, vs, True))
                self.state[p]["vs"] = vs
            self.state[p]["latent_p"] = old_p

    @staticmethod
    def quantize_param_(group, state, p_buf, p):
        """Quantize `p_buf` based on `group["quant_method"]`, saving into `p`."""
        quant_method = group["quant_method"]
        quant_cls = group["quant_cls"]
        if quant_method == "lsq":
            quant_target = p_buf.div(state["scale"])
            qmin, qmax = group["qmin"], group["qmax"]
            args = (quant_target, qmin, qmax)
            neg_mask, mid_mask, pos_mask = quant_cls.get_grad_masks(*args)

            grad_scale = quant_cls.get_grad_scale(
                *args, neg_mask, mid_mask, pos_mask, p.grad
            )
            state["scale"].sub_(grad_scale.sum(), alpha=group["lr"])

            p.copy_(quant_cls.apply(p_buf, state["scale"], qmin, qmax))
            p.grad.copy_(quant_cls.get_grad_target(p.grad, mid_mask))
        elif quant_method == "least-sq":
            p.copy_(quant_cls.apply(p_buf, state["vs"], True))
