import torch.nn as nn
from fairseq.utils import get_activation_fn
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class BottleneckAdapter(nn.Module):
    """
    A simple adapter piece specific to a language-pair
    """
    def __init__(self, in_dim, bottleneck_dim, activation="relu", dropout=0, normalize_before=False):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.activation_fn = get_activation_fn(activation=activation)

        self.layer_norm = nn.LayerNorm(self.in_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        
        self.normalize_before = normalize_before
        self.fc1 = nn.Linear(self.in_dim, self.bottleneck_dim)
        self.fc2 = nn.Linear(self.bottleneck_dim, self.in_dim)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x += residual
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
        

class BottleneckAdapterBlock(nn.Module):
    """
    A simple adapter block which houses mulitple mulitple adapter pieces, i.e. one per langauge-pair
    """
    def __init__(self, 
                 lang_ids, 
                 in_dim, 
                 bottleneck_dim,
                 activation="relu",
                 dropout=0,
                 normalize_before=False):

        super().__init__()
        self.lang_ids = lang_ids
        self.adapters = nn.ModuleDict({
            id: BottleneckAdapter(
                in_dim,
                bottleneck_dim,
                activation,
                dropout,
                normalize_before
            ) for id in self.lang_ids
        })

    def forward(self, x, lang_id):
        return self.adapters[lang_id](x)