import torch.nn as nn
from fairseq import utils
from fairseq.modules import MultiheadAttention


class FactorizedEmbedding(nn.Module):
    """
    Self Attentive Factorized Embedding from "SubFormer: "
    Args:
    """

    def __init__(
        self, 
        num_embeddings,
        safe_embed_dim=128,
        safe_num_heads=8,
        safe_dropout=0,
        padding_idx=1
    ):
        super().__init__()
        self.embedding_dim = safe_embedding_dim
        self.padding_idx = padding_idx

        self.emb = nn.Embedding(num_embeddings, hid_dim, padding_idx=padding_idx)
        self.activation_fn = utils.get_activation_fn(activation=activation)

    def forward(self, x):
        return self.up(self.activation_fn(self.emb(x)))