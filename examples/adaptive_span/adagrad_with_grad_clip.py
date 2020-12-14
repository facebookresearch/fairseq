# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import Adagrad

from fairseq.optim import LegacyFairseqOptimizer, register_optimizer


@register_optimizer("adagrad_with_grad_clip")
class FairseqAdagradWithGradClip(LegacyFairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = AdagradWithGradClip(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--adagrad-clip', default=0.0, type=float, metavar='D',
                            help='internal grad clip')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.args.lr[0],
            "weight_decay": self.args.weight_decay,
            "grad_clip": self.args.adagrad_clip,
        }

    @property
    def supports_flat_params(self):
        return False


def _clip_grad(clr, grad, group_grad_clip):
    if group_grad_clip > 0:
        norm = grad.norm(2).item()
        if norm > group_grad_clip:
            clr *= group_grad_clip / (norm + 1e-10)
    return clr


class AdagradWithGradClip(Adagrad):
    """Adagrad algorithm with custom gradient clipping"""

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        grad_clip=0,
    ):
        Adagrad.__init__(
            self,
            params,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )
        self.defaults["grad_clip"] = grad_clip
        self.param_groups[0].setdefault("grad_clip", grad_clip)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is "
                            "not compatible with sparse "
                            "gradients"
                        )
                    grad = grad.add(group["weight_decay"], p.data)

                clr = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])

                # clip
                clr = _clip_grad(clr=clr, grad=grad, group_grad_clip=group["grad_clip"])

                if grad.is_sparse:
                    # the update is non-linear so indices must be unique
                    grad = grad.coalesce()
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    state["sum"].add_(make_sparse(grad_values.pow(2)))
                    std = state["sum"]._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state["sum"].addcmul_(1, grad, grad)
                    std = state["sum"].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss
