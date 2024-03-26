import torch.nn as nn
from fairseq import utils


class FactorizedEmbedding(nn.Module):
    """
    Factorized Embedding from "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (Lan et al.,)
    <https://arxiv.org/abs/1909.11942>

    Args:
        num_embeddings: vocabulary size
        embedding_dim: Final embedding dimension
        hid_dim: factored lower dimension for embedding vectors
        padding_idx: pad token index in the vocabulary
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        hid_dim=128,
        padding_idx=1,
        bias=False,
        activation="linear",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.up = nn.Linear(hid_dim, embedding_dim, bias=bias)
        self.emb = nn.Embedding(num_embeddings, hid_dim, padding_idx=padding_idx)
        self.activation_fn = utils.get_activation_fn(activation=activation)

    def forward(self, x):
        return self.up(self.activation_fn(self.emb(x)))
