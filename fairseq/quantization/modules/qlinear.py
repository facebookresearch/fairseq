#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """
    Quantized counterpart of nn.Linear module. Stores the centroid, the assignments
    and the non-quantized biases. The full weight is re-instantiaced at each forward
    pass. For performance issues and contrary to the implementation of QuantizedConv2d,
    the backward is written by hand, since linear layers require less compute than
    convolutional layers for the same number of parameters.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_features x n_blocks
        - bias: the non-quantized bias

    Remarks:
        - We refer the reader to the official documentation of the nn.Linear module
          for the other arguments and the behavior of the module
        - Performance tests on GPU show that this implementation is 20% slower than
          the non-quantized nn.Linear module for a standard training loop.
    """

    def __init__(self, centroids, assignments, bias, in_features, out_features):
        super(QuantizedLinear, self).__init__()
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
        if torch.any(self.counts == 0):
            raise ValueError("Some centroids are not used")
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
        return QuantizedLinearFunction.apply(
            x,
            self.centroids,
            self.bias,
            self.assignments,
            self.in_features,
            self.out_features,
            self.block_size,
            self.counts,
        )

    def extra_repr(self):
        return f"in_features={self.in_features},\
                 out_features={self.out_features},\
                 n_centroids={self.n_centroids},\
                 block_size={self.block_size},\
                 bias={self.bias is not None}"


class QuantizedLinearFunction(torch.autograd.Function):
    """
    Autograd function defining the forward and the backward of the linear layer.
    Note that the gradients are averaged by cluster during the backward instead
    of being summed.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        centroids,
        bias,
        assignments,
        in_features,
        out_features,
        block_size,
        counts,
    ):
        weight = (
            centroids[assignments]
            .reshape(-1, out_features, block_size)
            .permute(1, 0, 2)
            .flatten(1, 2)
        )
        ctx.save_for_backward(x, weight, bias, centroids, assignments, counts)
        ctx.in_features = in_features
        ctx.block_size = block_size
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, centroids, assignments, counts = ctx.saved_tensors
        in_features, block_size = ctx.in_features, ctx.block_size
        n_blocks = in_features // block_size
        grad_input = grad_centroids = grad_bias = None

        # gradient wrt input x
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        # gradient wrt centroids
        if ctx.needs_input_grad[1]:
            grad_centroids = torch.zeros_like(centroids)
            grad_reshape = (
                x.reshape(-1, n_blocks, block_size)
                .permute(2, 1, 0)
                .matmul(grad_output)
                .flatten(1, 2)
                .t()
            )
            grad_centroids.index_add_(0, assignments, grad_reshape)
            grad_centroids /= counts[:, None]
        # gradient wrt bias
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_centroids, grad_bias, None, None, None, None, None
