import torch as th
from torch.optim.optimizer import Optimizer, required
from . import register_multiobj_optim


def normalize_param(W):
    return W / W.norm(2).clamp(min=1e-12)


def to_vector(tensors):
    """Flatten a list of parameters/gradients to a vector"""
    return th.cat([t.view(-1) for t in tensors])


def from_vector(tensors, vector):
    """Reverse `to_vector` (overwrites the tensor values)"""
    pointer = 0
    for tensor in tensors:
        new_val = vector[pointer:pointer+tensor.numel()].view(tensor.size())
        tensor.copy_(new_val)
        pointer += tensor.numel()


class MultiObjSGD(Optimizer):
    """
    This optimizer works like SGD excepts:

    1. it stores gradient from an auxiliary task with `.save_auxiliary()`
    2. it uses those auxiliary gradients using `.combine_gradientss()` before
        applying the update

    Args:
        full_gradients (bool): do gradient combination ops on the full
            gradients (as opposed to separately for each parameter)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, always_project=True,
                 full_gradients=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        frozen=False)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(MultiObjSGD, self).__init__(params, defaults)
        self.always_project = always_project
        self.full_gradients = full_gradients

    def __setstate__(self, state):
        super(MultiObjSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def save_auxiliary(self):
        """This saves the gradients wrt. the auxiliary objective"""

        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                # skip frozen parameters (TODO: remove this)
                if getattr(param_state, "frozen", False):
                    continue
                # Actually save the gradient
                param_state["aux_grad"] = th.zeros_like(p.data)
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state["aux_grad"].add_(d_p)

    def combine_gradients(self, g_p, aux_g_p):
        """Manipulate the gradient g_p using the gradient from the auxiliary
        objective aux_g_p"""
        raise NotImplementedError()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Apply momentum and everything to get final gradient
        params = []
        lrs = []
        grads = []
        aux_grads = []
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                # skip frozen parameters
                if getattr(param_state, "frozen", False):
                    print("Skipping parameter of size", p.dim())
                    continue
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = th.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # Track parameters, learning rate, gradients and auxiliary
                # gradients
                params.append(p)
                lrs.append(param_state["lr"])
                grads.append(d_p)
                if "aux_grad" in param_state:
                    aux_grads.append(param_state["aux_grad"])
                else:
                    aux_grads.append(th.zeros_like(d_p))

        # Combine gradients
        if self.full_gradients:
            # Consider parameters as one vector
            new_grad_vec = self.combine_gradients(
                to_vector(grads),
                to_vector(aux_grads)
            )
            # Overwrite gradients
            from_vector(grads, new_grad_vec)
        else:
            # Treat each parameter independently
            grads = [self.combine_gradients(g, aux_g)
                     for g, aux_g in zip(grads, aux_grads)]

        # Apply the update
        for p, lr, g in zip(params, lrs, grads):
            p.data.add_(-lr, g)

        return loss


@register_multiobj_optim("single")
class SingleObjSGD(MultiObjSGD):
    """Same as SGD ("single" objective)"""

    def combine_gradients(self, g_p, aux_g_p):
        return g_p


@register_multiobj_optim("avg")
class AvgMultiObjSGD(MultiObjSGD):
    """Average the gradients"""

    def combine_gradients(self, g_p, aux_g_p):
        avg_p = 0.5 * (aux_g_p + g_p)
        return avg_p


@register_multiobj_optim("ortho")
class OrthoMultiObjSGD(MultiObjSGD):
    """Project the gradient g_p on the hyperplane orthogonal to aux_g_p"""

    def combine_gradients(self, g_p, aux_g_p):
        c_unit = aux_g_p / (aux_g_p.norm(2) + 1e-10)
        dot = (g_p * c_unit).sum()
        # Only project if the gradients have negative dot product
        if self.always_project or dot.data <= 0:
            return g_p - dot * c_unit
        else:
            return g_p


@register_multiobj_optim("nullify")
class NullifyMultiObjSGD(MultiObjSGD):
    """Nullify the gradient if the directions are not aligned"""

    def combine_gradients(self, g_p, aux_g_p):
        if (g_p * aux_g_p).sum() <= 0:
            return th.zeros_like(g_p)
        else:
            return aux_g_p


@register_multiobj_optim("cwise-ortho")
class CwiseOrthoMultiObjSGD(MultiObjSGD):
    """Orthogonal projection but at the level of scalar parameters"""

    def combine_gradients(self, g_p, aux_g_p):
        mask = th.nn.functional.relu(th.sign(g_p * aux_g_p))
        return mask * g_p


@register_multiobj_optim("cosine-weighted")
class CosineWeightedMultiObjSGD(MultiObjSGD):
    """Weight the update by the (rectified) cosine similarity between the two
    gradients. Update in the direction of aux_g_p"""

    def combine_gradients(self, g_p, aux_g_p):
        c_unit = aux_g_p / (aux_g_p.norm(2) + 1e-10)
        g_unit = g_p / (g_p.norm(2) + 1e-10)
        cosine = (g_unit * c_unit).sum()
        return th.nn.functional.relu(cosine) * g_p


@register_multiobj_optim("cosine-weighted-sum")
class CosineWeightedSumMultiObjSGD(MultiObjSGD):
    """Weight the update by the (rectified) cosine similarity between the two
    gradients. Update in the direction of g_p + aux_g_p
    (see https://arxiv.org/abs/1812.02224)"""

    def combine_gradients(self, g_p, aux_g_p):
        c_unit = aux_g_p / (aux_g_p.norm(2) + 1e-10)
        g_unit = g_p / (g_p.norm(2) + 1e-10)
        cosine = (g_unit * c_unit).sum()
        return th.nn.functional.relu(cosine) * 0.5 * (g_p + aux_g_p)


@register_multiobj_optim("colinear")
class ColinearMultiObjSGD(MultiObjSGD):
    """Project g_p on the direction of aux_g_p (when the 2 are colinear)"""

    def combine_gradients(self, g_p, aux_g_p):
        c_unit = aux_g_p / (aux_g_p.norm(2) + 1e-10)
        dot = (c_unit * g_p).sum()
        return th.nn.functional.relu(dot) * c_unit


@register_multiobj_optim("same-contrib")
class SameContribMultiObjSGD(MultiObjSGD):
    """Here the update is a vector d such that
    Loss_1(x + d) - Loss_1(x) = Loss_2(x + d) - Loss_2(x)"""

    def combine_gradients(self, g_p, aux_g_p):
        diff = g_p - aux_g_p
        diff_norm = diff.norm(2) + 1e-10
        diff_unit = diff / diff_norm
        dot = (g_p * diff_unit).sum()
        return g_p - dot * diff_unit


@register_multiobj_optim("avg-ortho")
class AvgOrthoMultiObjSGD(MultiObjSGD):
    """Project g_p on the orthogonal of aux_g_p, and aux_g_p on the orthogonal
    of g_p, then average"""

    def combine_gradients(self, g_p, aux_g_p):
        g_norm = g_p.norm(2)+1e-10
        c_norm = aux_g_p.norm(2)+1e-10
        dot = (g_p * aux_g_p).sum()
        if self.always_project or dot.data <= 0:
            g_unit = g_p / g_norm
            c_unit = aux_g_p / c_norm
            g_proj_c = g_p - (g_p * c_unit).sum() * c_unit
            aux_g_proj_g = aux_g_p - (aux_g_p * g_unit).sum() * g_unit
            return 0.5 * (g_proj_c + aux_g_proj_g)
        else:
            # If the two are somewhat aligned, no need to project
            return g_p
