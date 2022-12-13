import torch.nn as nn
import torch.nn.functional as F

class SimpleAdapter(nn.Module):
    """
    A simple adapter piece specific to a language
    """
    def __init__(
        self, in_dim, red_factor=2, activation_fn="relu", dropout=0, normalize_before=False
    ):

        super().__init__()

        self.act = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "mish": F.mish,
            "leaky_relu": F.leaky_relu,
        }[activation_fn]

        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.normalize_before = normalize_before
        self.fc1 = nn.Linear(in_dim, in_dim//red_factor)
        self.fc2 = nn.Linear(in_dim//red_factor, in_dim)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x += residual
        if self.normalize_before:
            x = self.layer_norm(x)
        return x
        

class SimpleAdapterBlock(nn.Module):
    """
    A simple adapter block which houses mulitple mulitple adapter pieces, i.e. one per langauge
    """
    def __init__(
        self, lang_ids, in_dim, red_factor=2, activation_fn="relu", dropout=0, normalize_before=False
    ):
        super().__init__()

        self.adapters = nn.ModuleDict({
            id: SimpleAdapter(
                in_dim=in_dim,
                dropout=dropout,
                red_factor=red_factor,
                activation_fn=activation_fn,
                normalize_before=normalize_before,
            ) for id in lang_ids
        })

    def forward(self, x, lang_id):
        return self.adapters[lang_id](x)