# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Layer norm done in fp32 (for fp16 training)
"""

import torch.nn as nn
import torch.nn.functional as F


class Fp32InstanceNorm(nn.InstanceNorm1d):
    def __init__(self, *args, **kwargs):
        self.transpose_last = "transpose_last" in kwargs and kwargs["transpose_last"]
        if "transpose_last" in kwargs:
            del kwargs["transpose_last"]
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.transpose_last:
            input = input.transpose(1, 2)
        output = F.instance_norm(
            input.float(),
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight.float() if self.weight is not None else None,
            bias=self.bias.float() if self.bias is not None else None,
            use_input_stats=self.training or not self.track_running_stats,
            momentum=self.momentum,
            eps=self.eps,
        )
        if self.transpose_last:
            output = output.transpose(1, 2)
        return output.type_as(input)
