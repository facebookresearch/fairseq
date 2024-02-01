import torch
import torch.nn as nn


class AdaptedLinear(nn.Linear):
    def __init__(self, weight_init: torch.Tensor, bias_init: torch.Tensor):
        factory_kwargs = {"dtype": weight_init.dtype, "device": weight_init.device}
        in_features = weight_init.shape[1]
        out_features = weight_init.shape[0]
        super().__init__(
            in_features, out_features, bias=bias_init is not None, **factory_kwargs
        )

        self._weight = nn.Parameter(self.weight.detach().clone())
        delattr(self, "weight")

        if self.bias is not None:
            self.bias.data.copy_(bias_init.data)
