import torch
from torch.autograd import Variable, Function
from torch.nn.modules.utils import _single

try:
    from fairseq import temporal_convolution_tbc
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: missing temporal_convolution_tbc, run `python setup.py install`\n')
    raise e


class ConvTBCFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, pad):
        input_size = input.size()
        weight_size = weight.size()
        kernel_size = weight_size[0]

        output = input.new(
            input_size[0] - kernel_size + 1 + pad * 2,
            input_size[1],
            weight_size[2])

        ctx.input_size = input_size
        ctx.weight_size = weight_size
        ctx.save_for_backward(input, weight)
        temporal_convolution_tbc.TemporalConvolutionTBC_forward(
            input, output, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_output = grad_output.data.contiguous()
        grad_input = grad_output.new(ctx.input_size).zero_()
        grad_weight = grad_output.new(ctx.weight_size).zero_()
        grad_bias = grad_output.new(ctx.weight_size[2])

        temporal_convolution_tbc.TemporalConvolutionTBC_backward(
            grad_output,
            grad_input,
            grad_weight,
            grad_bias,
            input,
            weight)

        grad_input = Variable(grad_input, volatile=True)
        grad_weight = Variable(grad_weight, volatile=True)
        grad_bias = Variable(grad_bias, volatile=True)

        return grad_input, grad_weight, grad_bias, None


class ConvTBC(torch.nn.Module):
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
