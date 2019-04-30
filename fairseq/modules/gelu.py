# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
See "Gaussian Error Linear Units (GELUs)" by Dan Hendrycks and Kevin Gimpel with
the corresponding GitHub repo: https://github.com/hendrycks/GELUs
"""

import math

import torch


def gelu_fast(x):
    if not hasattr(gelu_fast, "_a"):
        gelu_fast._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_fast._a * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
