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
    """

    def __init__(self, num_embeddings, embedding_dim, hid_dim=128, padding_idx=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.em = nn.Embedding(num_embeddings, hid_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(hid_dim, embedding_dim)

    def forward(self, x):
        return self.fc(self.em(x))