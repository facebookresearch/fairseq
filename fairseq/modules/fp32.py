# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
batch norm done in fp32 (for fp16 training)
"""
import torch
import torch.nn as nn


class Fp32(nn.Module):
    def forward(self, input):
        return input.float()
