# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from fairseq import distributed_utils


class TPUDistributedDataParallel(nn.Module):

    def __init__(self, module, process_group):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = distributed_utils.get_world_size(self.process_group)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def all_reduce_grads(self):
        gradients = []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if p.grad.requires_grad:
                raise RuntimeError(
                    "TPUDistributedDataParallel only works with gradients that don't "
                    "require grad"
                )
            gradients.append(p.grad)

        import torch_xla.core.xla_model as xm
        xm.all_reduce(
            'sum',
            gradients,
            scale=1. / self.world_size,
            groups=self.process_group[1],
        )
