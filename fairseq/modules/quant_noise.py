import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter


class StructuredDropout(nn.Module):
    """
    Randomly drops blocks of columns in the input.
    Args:
        - p: dropout probability
        - block_size: size of the block of columns
    Remarks:
        - As in the standard dropout implementation, the input is scaled by
          a factor 1 / (1 - p) during training and left unchanged during evaluation.
    """

    def __init__(self, p, block_size):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p
        self.block_size = int(block_size)

    def forward(self, x):
        # training
        if self.training and self.p > 0:
            # generate mask
            # x is T x B x C
            bptt, bs, d = x.size()
            mask = torch.zeros(bs, 1, d // self.block_size, device=x.device)
            mask.bernoulli_(self.p)
            mask = mask.repeat_interleave(self.block_size, -1).bool()
            # scaling
            s = 1 / (1 - self.p)
            return s * x.masked_fill(mask, 0)
        # eval mode no dropout
        else:
            return x


class StructuredDropLinear(nn.Module):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    with possible block dropout inside the matrix
    """

    def __init__(self, in_features, out_features, bias=True, p=0, block_size=8):
        super(StructuredDropLinear, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.chosen_bias = bias
        if self.chosen_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.p = p
        self.block_size = int(block_size)
        if p > 0:
            assert in_features % block_size == 0, "in_features must be a multiple of block size"
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.chosen_bias:
            nn.init.constant_(self.bias, 0.)
        return

    def forward(self, input):
        # dropout on the blocks of the weight matrix
        if self.training and self.p > 0:
            mask = torch.zeros(int(self.in_features // self.block_size * self.out_features), device=self.weight.device)
            mask.bernoulli_(self.p)
            mask = mask.repeat_interleave(self.block_size, -1).view(-1, self.in_features).bool()
            s = 1 / (1 - self.p)
            weight =  s * self.weight.masked_fill(mask, 0)
            return F.linear(input, weight, self.bias)
        # eval mode no dropout
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, dropout={}, block_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.p, self.block_size
        )
