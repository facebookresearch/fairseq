import torch.nn as nn

class SimpleAdapterBlock(nn.Module):
    """
    This is a simple adapater block and it does not use quant_noise/quant_noise_block_size as used in xmod
    """
    def __init__(
        self, lang_ids, in_dim, red_factor=2, activation_fn="silu", dropout=0.1, normalize_before_adapter=False
    ):
        super().__init__()

        activation_fn_module = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "prelu": nn.PReLU(),
            "leaky_relu": nn.LeakyReLU(),
        }[activation_fn]

        self.adapters = nn.ModuleDict({
            id: nn.Sequential(
                nn.Linear(in_dim, in_dim//red_factor),
                activation_fn_module,
                nn.Linear(in_dim//red_factor, in_dim)
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
        