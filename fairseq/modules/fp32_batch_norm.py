# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
batch norm done in fp32 (for fp16 training)
"""
import torch
import torch.nn as nn


class Fp32BatchNorm(nn.Module):
    def __init__(self, sync=False, *args, **kwargs):
        super().__init__()

        if sync:
            from fairseq.distributed import utils

            if utils.get_global_world_size() == 1:
                sync = False

        if sync:
            self.bn = nn.SyncBatchNorm(*args, **kwargs)
        else:
            self.bn = nn.BatchNorm1d(*args, **kwargs)

        self.sync = sync

    def forward(self, input):
        if self.bn.running_mean.dtype != torch.float:
            if self.sync:
                self.bn.running_mean = self.bn.running_mean.float()
                self.bn.running_var = self.bn.running_var.float()
                if self.bn.affine:
                    try:
                        self.bn.weight = self.bn.weight.float()
                        self.bn.bias = self.bn.bias.float()
                    except:
                        self.bn.float()
            else:
                self.bn.float()

        output = self.bn(input.float())
        return output.type_as(input)
