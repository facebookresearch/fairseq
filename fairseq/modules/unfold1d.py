# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn.functional as F


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    '''unfold T x B x C to T x B x C x K'''
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B*C, C, 1, B*C))
    else:
        x = x.unsqueeze(3)
    return x
