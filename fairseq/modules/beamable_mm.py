# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch
import torch.nn as nn


class BeamableMM(nn.Module):
    """This module provides an optimized MM for beam decoding with attention.

    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds up
    inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """
    def __init__(self, beam_size=None):
        super(BeamableMM, self).__init__()
        self.beam_size = beam_size

    def forward(self, input1, input2):
        if (
            not self.training and           # test mode
            self.beam_size is not None and  # beam size is set
            input1.dim() == 3 and           # only support batched input
            input1.size(1) == 1             # single time step update
        ):
            bsz, beam = input1.size(0), self.beam_size

            # bsz x 1 x nhu --> bsz/beam x beam x nhu
            input1 = input1[:, 0, :].unfold(0, beam, beam).transpose(2, 1)

            # bsz x sz2 x nhu --> bsz/beam x sz2 x nhu
            input2 = input2.unfold(0, beam, beam)[:, :, :, 0]

            # use non batched operation if bsz = beam
            if input1.size(0) == 1:
                output = torch.mm(input1[0, :, :], input2[0, :, :])
            else:
                output = input1.bmm(input2)
            return output.view(bsz, 1, -1)
        else:
            return input1.bmm(input2)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size
