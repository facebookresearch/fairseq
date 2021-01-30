# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PQLinear(nn.Module):
    """
    Quantized counterpart of nn.Linear module. Stores the centroid, the assignments
    and the non-quantized biases. The full weight is re-instantiated at each forward
    pass.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_features x n_blocks
        - bias: the non-quantized bias

    Remarks:
        - We refer the reader to the official documentation of the nn.Linear module
          for the other arguments and the behavior of the module
        - Performance tests on GPU show that this implementation is 15% slower than
          the non-quantized nn.Linear module for a standard training loop.
    """

    def __init__(self, centroids, assignments, bias, in_features, out_features):
        super(PQLinear, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.in_features = in_features
        self.out_features = out_features
        # check compatibility
        if self.in_features % self.block_size != 0:
            raise ValueError("Wrong PQ sizes")
        if len(assignments) % self.out_features != 0:
            raise ValueError("Wrong PQ sizes")
        # define parameters
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer("assignments", assignments)
        self.register_buffer("counts", torch.bincount(assignments).type_as(centroids))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        return (
            self.centroids[self.assignments]
            .reshape(-1, self.out_features, self.block_size)
            .permute(1, 0, 2)
            .flatten(1, 2)
        )

    def forward(self, x):
        return F.linear(
            x,
            self.weight,
            self.bias,
        )

    def extra_repr(self):
        return f"in_features={self.in_features},\
                 out_features={self.out_features},\
                 n_centroids={self.n_centroids},\
                 block_size={self.block_size},\
                 bias={self.bias is not None}"
