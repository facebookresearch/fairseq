# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch
from torch.autograd import Function
from torch.nn.modules.utils import _single

from fairseq import utils

try:
    from fairseq import temporal_convolution_tbc
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: missing temporal_convolution_tbc, run `python setup.py install`\n')
    raise e


class ConvTBC(torch.nn.Module):
    """1D convolution over an input of shape (time x batch x channel)

    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        assert self.stride == (1,)

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size[0], in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return ConvTBCFunction.apply(
            input.contiguous(), self.weight, self.bias, self.padding[0])

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', padding={padding}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvTBCFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, pad):
        input_size = input.size()
        weight_size = weight.size()
        kernel_size = weight_size[0]

        output = input.new(
            input_size[0] - kernel_size + 1 + int(pad * 2),
            input_size[1],
            weight_size[2])

        ctx.input_size = input_size
        ctx.weight_size = weight_size
        ctx.save_for_backward(input, weight)
        temporal_convolution_tbc.TemporalConvolutionTBC_forward(
            input.type().encode('utf-8'),
            input,
            output,
            weight,
            bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_output = grad_output.data.contiguous()
        grad_input = grad_output.new(ctx.input_size).zero_()
        grad_weight = grad_output.new(ctx.weight_size).zero_()
        grad_bias = grad_output.new(ctx.weight_size[2])

        temporal_convolution_tbc.TemporalConvolutionTBC_backward(
            input.type().encode('utf-8'),
            grad_output,
            grad_input,
            grad_weight,
            grad_bias,
            input,
            weight)

        grad_input = utils.volatile_variable(grad_input)
        grad_weight = utils.volatile_variable(grad_weight)
        grad_bias = utils.volatile_variable(grad_bias)

        return grad_input, grad_weight, grad_bias, None


def conv_tbc(input, weight, bias=None, stride=1, padding=0):
    return ConvTBCFunction.apply(
        input.contiguous(), weight, bias, padding[0])
