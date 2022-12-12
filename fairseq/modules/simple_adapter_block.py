import torch.nn as nn

class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, red_factor=2, activation_fn="relu"):
        super().__init__()

        self.activation_fn_module = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "prelu": nn.PReLU(),
            "leaky_relu": nn.LeakyReLU(),
        }[activation_fn]

        self.fc1 = nn.Linear(in_dim, in_dim//red_factor)
        self.fc2 = nn.Linear(in_dim//red_factor, in_dim)
        self.reset_parameters_()

    def reset_parameters_(self):
        nn.init.normal_(self.fc1.weight.data, mean=0, std=0.02)
        nn.init.normal_(self.fc2.weight.data, mean=0, std=0.02)
        nn.init.constant_(self.fc1.bias.data, 0)
        nn.init.constant_(self.fc2.bias.data, 0)

    def forward(self, x):
        return self.fc2(self.activation_fn_module(self.fc1(x)))


class SimpleAdapterBlock(nn.Module):
    """
    This is a simple adapater block and it does not use quant_noise/quant_noise_block_size as used in xmod
    """
    def __init__(
        self, lang_ids, in_dim, red_factor=2, activation_fn="relu", dropout=0.1, normalize_before_adapter=False
    ):
        super().__init__()

        self.adapters = nn.ModuleDict({
            id: SimpleAdapter(
                in_dim, 
                red_factor, 
                activation_fn
            ) for id in lang_ids
        })
        self.normalize_before_adapter = normalize_before_adapter
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(p=dropout)

    def residual_connection(self, x, residual):
        return x + residual

    def forward(self, x, lang_id):
        if not self.normalize_before_adapter:
            residual = x
        x = self.layer_norm(x)
        if self.normalize_before_adapter:
            residual = x
        x = self.adapters[lang_id](x)
        x = self.dropout(x)
        x = self.residual_connection(x, residual)
        return x
        