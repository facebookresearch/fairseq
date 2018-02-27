# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn.functional as F

from fairseq import utils

from .conv_tbc import ConvTBC


class LinearizedConvolution(ConvTBC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._is_incremental_eval = False
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def remove_future_timesteps(self, x):
        """Remove future time steps created by padding."""
        if not self._is_incremental_eval and self.kernel_size[0] > 1 and self.padding[0] > 0:
            x = x[:-self.padding[0], :, :]
        return x

    def incremental_eval(self, mode=True):
        self._is_incremental_eval = mode
        if mode:
            self.clear_incremental_state()

    def forward(self, input):
        if self._is_incremental_eval:
            return self.incremental_forward(input)
        else:
            return super().forward(input)

    def incremental_forward(self, input):
        """Forward convolution one time step at a time.

        This function maintains an internal state to buffer signal and accepts
        a single frame as input. If the input order changes between time steps,
        call reorder_incremental_state. To apply to fresh inputs, call
        clear_incremental_state.
        """
        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]

        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, kw, input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = utils.volatile_variable(self.input_buffer)
        with utils.maybe_no_grad():
            output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)

    def clear_incremental_state(self):
        self.input_buffer = None

    def reorder_incremental_state(self, new_order):
        if self.input_buffer is not None:
            self.input_buffer = self.input_buffer.index_select(0, new_order)

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None
