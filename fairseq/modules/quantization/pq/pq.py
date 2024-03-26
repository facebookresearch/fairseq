# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .em import EM, EmptyClusterResolveError


class PQ(EM):
    """
    Quantizes the layer weights W with the standard Product Quantization
    technique. This learns a codebook of codewords or centroids of size
    block_size from W. For further reference on using PQ to quantize
    neural networks, see "And the Bit Goes Down: Revisiting the Quantization
    of Neural Networks", Stock et al., ICLR 2020.

    PQ is performed in two steps:
    (1) The matrix W (weights or fully-connected or convolutional layer)
        is reshaped to (block_size, -1).
            - If W is fully-connected (2D), its columns are split into
              blocks of size block_size.
            - If W is convolutional (4D), its filters are split along the
              spatial dimension.
    (2) We apply the standard EM/k-means algorithm to the resulting reshaped matrix.

    Args:
        - W: weight matrix to quantize of size (in_features x out_features)
        - block_size: size of the blocks (subvectors)
        - n_centroids: number of centroids
        - n_iter: number of k-means iterations
        - eps: for cluster reassignment when an empty cluster is found
        - max_tentatives for cluster reassignment when an empty cluster is found
        - verbose: print information after each iteration

    Remarks:
        - block_size be compatible with the shape of W
    """

    def __init__(
        self,
        W,
        block_size,
        n_centroids=256,
        n_iter=20,
        eps=1e-6,
        max_tentatives=30,
        verbose=True,
    ):
        self.block_size = block_size
        W_reshaped = self._reshape(W)
        super(PQ, self).__init__(
            W_reshaped,
            n_centroids=n_centroids,
            n_iter=n_iter,
            eps=eps,
            max_tentatives=max_tentatives,
            verbose=verbose,
        )

    def _reshape(self, W):
        """
        Reshapes the matrix W as expained in step (1).
        """

        # fully connected: by convention the weight has size out_features x in_features
        if len(W.size()) == 2:
            self.out_features, self.in_features = W.size()
            assert (
                self.in_features % self.block_size == 0
            ), "Linear: n_blocks must be a multiple of in_features"
            return (
                W.reshape(self.out_features, -1, self.block_size)
                .permute(2, 1, 0)
                .flatten(1, 2)
            )

        # convolutional: we reshape along the spatial dimension
        elif len(W.size()) == 4:
            self.out_channels, self.in_channels, self.k_h, self.k_w = W.size()
            assert (
                self.in_channels * self.k_h * self.k_w
            ) % self.block_size == 0, (
                "Conv2d: n_blocks must be a multiple of in_channels * k_h * k_w"
            )
            return (
                W.reshape(self.out_channels, -1, self.block_size)
                .permute(2, 1, 0)
                .flatten(1, 2)
            )
        # not implemented
        else:
            raise NotImplementedError(W.size())

    def encode(self):
        """
        Performs self.n_iter EM steps.
        """

        self.initialize_centroids()
        for i in range(self.n_iter):
            try:
                self.step(i)
            except EmptyClusterResolveError:
                break

    def decode(self):
        """
        Returns the encoded full weight matrix. Must be called after
        the encode function.
        """

        # fully connected case
        if "k_h" not in self.__dict__:
            return (
                self.centroids[self.assignments]
                .reshape(-1, self.out_features, self.block_size)
                .permute(1, 0, 2)
                .flatten(1, 2)
            )

        # convolutional case
        else:
            return (
                self.centroids[self.assignments]
                .reshape(-1, self.out_channels, self.block_size)
                .permute(1, 0, 2)
                .reshape(self.out_channels, self.in_channels, self.k_h, self.k_w)
            )
