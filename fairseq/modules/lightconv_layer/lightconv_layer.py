# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

import lightconv_cuda
from fairseq import utils


class lightconvFunction(Function):

    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l
        outputs = lightconv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = lightconv_cuda.backward(
                grad_output.contiguous(),
                ctx.padding_l,
                *ctx.saved_variables)
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None


class LightconvLayer(nn.Module):
    def __init__(
            self,
            input_size,
            kernel_size=1,
            padding_l=None,
            weight_softmax=False,
            num_heads=1,
            weight_dropout=0.,
            bias=False):
        super(LightconvLayer, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout = weight_dropout

        self.weight = nn.Parameter(torch.Tensor(num_heads, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.reset_parameters()

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        for k, v in state_dict.items():
            if k.endswith(prefix + 'weight'):
                if v.dim() == 3 and v.size(1) == 1:
                    state_dict[k] = v.squeeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, incremental_state=None):

        # during inference time, incremental BMM is faster
        if incremental_state is not None:
            T, B, C = x.size()
            K, H = self.kernel_size, self.num_heads
            R = C // H
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :, -self.kernel_size+1:])
            x_unfold = x_unfold.view(T*B*H, R, -1)

            weight = self.weight
            if self.weight_softmax:
                weight = F.softmax(weight.float(), dim=1).type_as(weight)

            weight = weight[:, -x_unfold.size(2):]

            K = weight.size(1)

            weight = weight.view(1, H, K).expand(T*B, H, K).contiguous().view(T*B*H, K, 1)

            weight = F.dropout(weight, self.weight_dropout, training=self.training)
            output = torch.bmm(x_unfold, weight)  # T*B*H x R x 1
            output = output.view(T, B, C)
            return output

        # during training time, use CUDA kernel
        else:
            x = x.permute(1, 2, 0).contiguous()
            weight = self.weight
            if self.weight_softmax:
                weight = F.softmax(self.weight, -1)
            if self.weight_dropout:
                weight = F.dropout(weight, self.weight_dropout, training=self.training)
            return lightconvFunction.apply(x, weight, self.padding_l).permute(2, 0, 1)

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)
