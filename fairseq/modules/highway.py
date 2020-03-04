# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import nn


class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_.
    Adopted from the AllenNLP implementation.
    """

    def __init__(
            self,
            input_dim: int,
            num_layers: int = 1
    ):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(num_layers)])
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            # As per comment in AllenNLP:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            nn.init.constant_(layer.bias[self.input_dim:], 1)

            nn.init.constant_(layer.bias[:self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(
            self,
            x: torch.Tensor
    ):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
        return x
