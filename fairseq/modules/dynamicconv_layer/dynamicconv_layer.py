# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

import dynamicconv_cuda
from fairseq import utils
from fairseq.modules.unfold import unfold1d
from fairseq.incremental_decoding_utils import with_incremental_state


class dynamicconvFunction(Function):

    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l
        outputs = dynamicconv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = dynamicconv_cuda.backward(
                grad_output.contiguous(),
                ctx.padding_l,
                *ctx.saved_tensors)
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None


@with_incremental_state
class DynamicconvLayer(nn.Module):
    def __init__(
            self,
            input_size,
            kernel_size=1,
            padding_l=None,
            weight_softmax=False,
            num_heads=1,
            weight_dropout=0.,
            bias=False,
            renorm_padding=False,
            conv_bias=False,
            query_size=None):

        super(DynamicconvLayer, self).__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout = weight_dropout
        self.renorm_padding = renorm_padding
        self.bias = bias

        self.weight_linear = nn.Linear(input_size, num_heads * kernel_size, bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight)
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)
            nn.init.constant_(self.weight_linaer.bias, 0.)

    def forward(self, x, incremental_state=None, query=None, unfold=None):

        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        # R = C // H

        # during inference time, incremental BMM is faster
        if incremental_state is not None:
            unfold = x.size(0) > 512 if unfold is None else unfold  # use unfold mode as default for long sequence to save memory
            unfold = unfold or (incremental_state is not None)
            assert query is None

            if query is None:
                query = x
            if unfold:
                output = self._forward_unfolded(x, incremental_state, query)
            else:
                output = self._forward_expanded(x, incremental_state, query)

            if self.conv_bias is not None:
                output = output + self.conv_bias.view(1, 1, -1)

            return output

        # during training time, use CUDA kernel
        else:
            weight = self.weight_linear(x).view(T, B, H, K)
            if self.weight_softmax:
                weight = F.softmax(weight, dim=-1)
            if self.weight_dropout:
                weight = F.dropout(weight, self.weight_dropout, training=self.training)

            weight = weight.permute(1, 2, 3, 0).contiguous()
            self.filters = weight
            x = x.permute(1, 2, 0).contiguous()
            output = dynamicconvFunction.apply(x, weight, self.padding_l).permute(2, 0, 1)
            if self.conv_bias is not None:
                output = output + self.conv_bias.view(1, 1, -1)
            return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def _forward_unfolded(self, x, incremental_state, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight_linear(query).view(T*B*H, -1)

        # renorm_padding is only implemented in _forward_expanded
        assert not self.renorm_padding or incremental_state is not None

        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :, -self.kernel_size+1:])
            x_unfold = x_unfold.view(T*B*H, R, -1)
        else:
            padding_l = self.padding_l
            if K > T and padding_l == K-1:
                weight = weight.narrow(1, K-T, T)
                K, padding_l = T, T-1
            # unfold the input: T x B x C --> T' x B x C x K
            x_unfold = unfold1d(x, K, padding_l, 0)
            x_unfold = x_unfold.view(T*B*H, R, K)

        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, K)

        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)

        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)

        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)

        output = torch.bmm(x_unfold, weight.unsqueeze(2))  # T*B*H x R x 1
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, incremental_stat, query):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        '''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight_linear(query).view(T*B*H, -1)

        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)

        x = x.view(T, B*H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            # turn the convolution filters into band matrices
            weight_expanded = weight.new(B*H, T, T+K-1).fill_(float('-inf'))
            weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            # normalize the weight over valid positions like self-attention
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training, inplace=False)
        else:
            P = self.padding_l
            # For efficieny, we cut the kernel size and reduce the padding when the kernel is larger than the length
            if K > T and P == K-1:
                weight = weight.narrow(2, K-T, T)
                K, P = T, T-1
            # turn the convolution filters into band matrices
            weight_expanded = weight.new_zeros(B*H, T, T+K-1, requires_grad=False)
            weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)  # B*H x T x T
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output
