import torch
import torch.nn as nn
from fairseq import utils


class BottleneckAdapter(nn.Module):
    """
    A simple adapter piece specific to a language-pair
    """
    def __init__(self, 
        in_dim=512, 
        reduction_factor=16, 
        activation="relu", 
        dropout=0, 
        ln_before=False, 
        use_gating=False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = in_dim // reduction_factor
        
        self.use_gating = use_gating
        
        self.ln_before = ln_before
        self.layer_norm = nn.LayerNorm(self.in_dim)
        self.dropout_module = nn.Dropout(p=dropout)

        self.up = nn.Linear(self.in_dim, self.bottleneck_dim)
        self.down = nn.Linear(self.bottleneck_dim, self.in_dim)
        self.activation_fn = utils.get_activation_fn(activation)

        if self.use_gating:
            self.gate = nn.Linear(self.in_dim, 1)   
            
    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x):
        residual = x
        if self.ln_before:
            x = self.layer_norm(x)

        x = self.down(self.activation_fn(self.up(x)))

        if self.use_gating:
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            x = x * gate

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.ln_before:
            x = self.layer_norm(x)

        return x
        


class BottleneckAdapterBlock(nn.Module):
    """
    A simple adapter block which houses mulitple adapter pieces, 
    i.e. one per language-pair
    """
    def __init__(self, 
                 adapter_ids, 
                 in_dim=512,
                 reduction_factor=16,
                 activation="relu",
                 dropout=0,
                 ln_before=False,
                 use_gating=False):

        super().__init__()
        self.adapters = nn.ModuleDict({
            id_: BottleneckAdapter(
                in_dim=in_dim,
                reduction_factor=reduction_factor,
                activation=activation,
                dropout=dropout,
                ln_before=ln_before,
                use_gating=use_gating
            ) for id_ in adapter_ids
        })

    def forward(self, x, adapter_id):
        return self.adapters[adapter_id](x)