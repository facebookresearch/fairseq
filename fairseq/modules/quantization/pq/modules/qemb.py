# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PQEmbedding(nn.Module):
    """
    Quantized counterpart of nn.Embedding module. Stores the centroids and
    the assignments. The full weight is re-instantiated at each forward
    pass.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_features x n_blocks
        - bias: the non-quantized bias

    Remarks:
        - We refer the reader to the official documentation of the nn.Embedding module
          for the other arguments and the behavior of the module
        - Performance tests on GPU show that this implementation is 10% slower than
          the non-quantized nn.Embedding module for a standard training loop.
    """

    def __init__(
        self,
        centroids,
        assignments,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
    ):
        super(PQEmbedding, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        # check compatibility
        if self.embedding_dim % self.block_size != 0:
            raise ValueError("Wrong PQ sizes")
        if len(assignments) % self.num_embeddings != 0:
            raise ValueError("Wrong PQ sizes")
        # define parameters
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer("assignments", assignments)
        self.register_buffer("counts", torch.bincount(assignments).type_as(centroids))

    @property
    def weight(self):
        return (
            self.centroids[self.assignments]
            .reshape(-1, self.num_embeddings, self.block_size)
            .permute(1, 0, 2)
            .flatten(1, 2)
        )

    def forward(self, input):
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self):
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        s += ", n_centroids={n_centroids}, block_size={block_size}"

        return s.format(**self.__dict__)
