# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from torch.optim.optimizer import Optimizer, required


class NAG(Optimizer):
    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, lr_old=lr, momentum=momentum, weight_decay=weight_decay)
        super(NAG, self).__init__(params, defaults)

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

                d_p = p.grad.data
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = d_p.clone().zero_()

                buf = param_state['momentum_buffer']

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(momentum * momentum * lr_correct, buf)
                p.data.add_(-(1 + momentum) * lr, d_p)

                buf.mul_(momentum * lr_correct).add_(-lr, d_p)

            group['lr_old'] = lr

        return loss
