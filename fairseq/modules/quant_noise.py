# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights
    as described in 'Training with Quantization Noise for Extreme Model Compression'
    """
    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    def _forward_pre_hook(mod, input):
        if mod.training:
            weight = mod.weight
            in_features = weight.size(0)
            out_features = weight.size(1)

            # split weight matrix into blocks and randomly drop selected blocks
            mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
            mask.bernoulli_(p)
            mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            # workaround: x.bool() is not currently supported in TorchScript
            mask = mask.to(torch.bool)
            s = 1 / (1 - p)

            mod.weight.data =  s * weight.masked_fill(mask.t(), 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module
