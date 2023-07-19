import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """An implementation of RMS Normalization.

    [Zhang and Sennrich](https://openreview.net/pdf?id=SygkZ3MTJE)
    """

    def __init__(
        self, dimension: int, epsilon: float = 1e-8, is_bias: bool = False
    ):
        """
        Args:
            dimension: the dimension of the layer output to normalize
            epsilon: an epsilon to prevent dividing by zero
                in case the layer has zero variance. (default = 1e-8)
            is_bias: a boolean value whether to include bias term
                while normalization
        """
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.is_bias = is_bias
        self.scale = nn.Parameter(torch.ones(self.dimension))

        if self.is_bias:
            self.bias = nn.Parameter(torch.zeros(self.dimension))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
        """
        x_std = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        x_norm = x / (x_std + self.epsilon)
        
        return ((self.scale * x_norm) + self.bias) if self.is_bias else (self.scale * x_norm)