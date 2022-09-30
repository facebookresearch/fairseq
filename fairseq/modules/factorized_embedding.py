# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

class FactorizedEmbedding(nn.Module):
    """
    Factorized Embedding from "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (Lan et al.,) 
    <https://arxiv.org/abs/1909.11942>

    Args:
        num_embeddings: vocabulary size
        embedding_dim: Final embedding dimension
        hid_dim: factored lower dimension for embedding vectors
        padding_idx: pad token index in the vocabulary
        layernorm: whether to normalize embedding vectors or not (default False)
    """

    def __init__(self, num_embeddings, embedding_dim, hid_dim, padding_idx, layernorm=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.em = nn.Embedding(num_embeddings, hid_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(hid_dim, embedding_dim, bias=False)
        self.layernorm = nn.LayerNorm(embedding_dim) if layernorm else None
        self._initialize_embed_params()

    def _initialize_embed_params(self):
        nn.init.normal_(self.em.weight, mean=0, std=self.embedding_dim**-0.5)
        nn.init.constant_(self.em.weight[self.padding_idx], 0)

    def forward(self, x):
        x = self.fc(self.em(x))
        return x if self.layernorm is None else self.layernorm(x)