# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter


def structured_dropout(module, p, block_size):
    def _forward_pre_hook(mod, input):
        if mod.training and p > 0:
            weight = mod.weight
            in_features = weight.size(0)
            out_features = weight.size(1)

            mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
            mask.bernoulli_(p)
            mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            # workaround: x.bool() is not currently supported in TorchScript
            mask = mask.to(torch.bool)
            s = 1 / (1 - p)

            mod.weight.data =  s * weight.masked_fill(mask.t(), 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

class StructuredDropout(nn.Module):
    """
    Randomly drops blocks of columns in the input.
    Args:
        - p: dropout probability
        - block_size: size of the block of columns
    Remarks:
        - As in the standard dropout implementation, the input is scaled by
          a factor 1 / (1 - p) during training and left unchanged during evaluation.
    """

    def __init__(self, p, block_size):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p
        self.block_size = int(block_size)

    def forward(self, x):
        # training
        if self.training and self.p > 0:
            # generate mask
            # x is T x B x C
            bptt, bs, d = x.size()
            mask = torch.zeros(bs, 1, d // self.block_size, device=x.device)
            mask.bernoulli_(self.p)
            mask = mask.repeat_interleave(self.block_size, -1).bool()
            # scaling
            s = 1 / (1 - self.p)
            return s * x.masked_fill(mask, 0)
        # eval mode no dropout
        else:
            return x
