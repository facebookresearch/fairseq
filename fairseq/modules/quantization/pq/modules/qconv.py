# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class PQConv2d(nn.Module):
    """
    Quantized counterpart of nn.Conv2d module. Stores the centroid, the assignments
    and the non-quantized biases. The full weight is re-instantiated at each forward
    pass and autograd automatically computes the gradients with respect to the
    centroids.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_channels x n_blocks
        - bias: the non-quantized bias, must be either torch.Tensor or None

    Remarks:
        - We refer the reader to the official documentation of the nn.Conv2d module
          for the other arguments and the behavior of the module.
        - Performance tests on GPU show that this implementation is 10% slower than
          the non-quantized nn.Conv2d module for a standard training loop.
        - During the backward, the gradients are averaged by cluster and not summed.
          This explains the hook registered to the centroids.
    """

    def __init__(
        self,
        centroids,
        assignments,
        bias,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
    ):
        super(PQConv2d, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        # check compatibility
        if in_channels // groups * np.prod(self.kernel_size) % self.block_size != 0:
            raise ValueError("Wrong PQ sizes")
        if len(assignments) % out_channels != 0:
            raise ValueError("Wrong PQ sizes")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        # define parameters
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer("assignments", assignments)
        self.register_buffer("counts", torch.bincount(assignments).type_as(centroids))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)
        # register hook for averaging gradients per centroids instead of summing
        self.centroids.register_hook(lambda x: x / self.counts[:, None])

    @property
    def weight(self):
        return (
            self.centroids[self.assignments]
            .reshape(-1, self.out_channels, self.block_size)
            .permute(1, 0, 2)
            .reshape(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            )
        )

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        s += ", n_centroids={n_centroids}, block_size={block_size}"
        return s.format(**self.__dict__)
