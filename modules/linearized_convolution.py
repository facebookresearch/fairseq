import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearizedConvolution(nn.Module):
    """This module performs temporal convolution one time step at a time.

    It maintains an internal state to buffer signal and accepts a single frame
    as input. This module is for forward evaluation **only** and does not
    support backpropagation.
    """

    def __init__(self, conv_module):
        super(LinearizedConvolution, self).__init__()
        assert isinstance(conv_module, nn.Conv1d), \
            'input conv_module must be an nn.Conv1d module'
        self.conv_module = conv_module
        self.clear_buffer()

    def clear_buffer(self):
        self.input_buffer = None
        self.weight, self.bias = None, None

    def reorder_buffer(self, new_order):
        if self.input_buffer is not None:
            self.input_buffer = self.input_buffer.index_select(0, new_order)

    def forward(self, input):
        if not input.volatile:
            raise RuntimeError('LinearizedConvolution only supports inference')

        # run forward pre hooks (e.g., weight norm)
        for hook in self.conv_module._forward_pre_hooks.values():
            hook(self.conv_module, input)

        # extract weight, bias and kw from Conv1d layer
        weight, bias = self._get_weight(), self._get_bias()
        kw = self.conv_module.weight.size(2)

        input = input.data
        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, kw, input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
        input = torch.autograd.Variable(input, volatile=True)
        output = F.linear(input.view(bsz, -1), weight, bias)
        return output.view(bsz, 1, -1)

    def _get_weight(self):
        if self.weight is None:
            nout, nin, kw = self.conv_module.weight.size()
            self.weight = self.conv_module.weight \
                .transpose(1, 2).contiguous() \
                .view(nout, kw * nin)
        return self.weight

    def _get_bias(self):
        if self.bias is None:
            self.bias = self.conv_module.bias
        return self.bias
