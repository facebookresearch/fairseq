# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch


def gelu(x):
    if not hasattr(gelu, '_a'):
        gelu._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu._a * (x + 0.044715 * torch.pow(x, 3))))
