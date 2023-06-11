import torch
import torch.nn as nn
from fairseq import utils


class BottleneckAdapter(nn.Module):
    """
    A simple adapter piece specific to a language-pair
    """
    def __init__(self, 
        in_dim, 
        bottleneck_dim, 
        activation="relu", 
        dropout=0, 
        ln_before=False, 
        use_gating=False,
        original_dropout=0,
        original_ln_before=False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_gating = self.use_gating
        self.activation_fn = utils.get_activation_fn(activation)
        
        self.layer_norm = nn.LayerNorm(self.in_dim)
        self.dropout_module = nn.Dropout(p=dropout)

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.bottleneck_dim),
            self.activation_fn(x),
            nn.Linear(self.bottleneck_dim, self.in_dim)
        )

        if self.use_gating:
            self.gate = nn.Linear(self.in_dim, 1)   

        self.original_layer_norm = nn.LayerNorm(self.in_dim)
        self.original_dropout_module = nn.Dropout(p=original_dropout)

    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x, original_residual):
        residual = x
        if self.ln_before:
            x = self.layer_norm(x)

        x = self.layers(x)

        if self.use_gating:
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            x *= gate

        x = self.dropout_module(x)
        x = self.residual(x, residual)
        if not self.ln_before:
            x = self.layer_norm(x)

        x = self.original_dropout_module(x)
        x = self.residual_connection(x, original_residual)
        if not self.original_ln_before:
            x = self.original_layer_norm(x)

        return x
        

class BottleneckAdapterBlock(nn.Module):
    """
    A simple adapter block which houses mulitple mulitple adapter pieces, 
    i.e. one per language-pair
    """
    def __init__(self, 
                 lang_ids, 
                 in_dim, 
                 bottleneck_dim,
                 activation="relu",
                 dropout=0,
                 ln_before=False,
                 use_gating=False,
                 original_dropout=0,
                 original_ln_before=False):

        super().__init__()
        self.lang_ids = lang_ids
        self.adapters = nn.ModuleDict({
            id: BottleneckAdapter(
                in_dim=in_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                dropout=dropout,
                ln_before=ln_before,
                original_dropout=original_dropout,
                original_ln_before=original_ln_before
            ) for id in self.lang_ids
        })

    def forward(self, x, lang_id, original_residual):
        return self.adapters[lang_id](x, original_residual)