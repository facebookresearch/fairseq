# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer, required

from . import FairseqOptimizer, register_optimizer


@register_optimizer('nag')
class FairseqNAG(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = NAG(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                            help='momentum factor')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
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
            'lr': self.args.lr[0],
            'momentum': self.args.momentum,
            'weight_decay': self.args.weight_decay,
        }


class NAG(Optimizer):
    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, lr_old=lr, momentum=momentum, weight_decay=weight_decay)
        super(NAG, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            lr_old = group.get('lr_old', lr)
            lr_correct = lr / lr_old

            for p in group['params']:
                if p.grad is None:
                    continue

                p_data_fp32 = p.data.float()

                d_p = p.grad.data.float()
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(d_p)
                else:
                    param_state['momentum_buffer'] = param_state['momentum_buffer'].type_as(d_p)

                buf = param_state['momentum_buffer']

                if weight_decay != 0:
                    p_data_fp32.mul_(1 - lr * weight_decay)
                p_data_fp32.add_(buf, alpha=momentum * momentum * lr_correct)
                p_data_fp32.add_(d_p, alpha=-(1 + momentum) * lr)

                buf.mul_(momentum * lr_correct).add_(d_p, alpha=-lr)

                # TODO: remove check once pyTorch avoids a copy for this case
                if p.data_ptr() != p_data_fp32.data_ptr():
                    p.data.copy_(p_data_fp32)

            group['lr_old'] = lr

        return loss
