# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
See "Gaussian Error Linear Units (GELUs)" by Dan Hendrycks and Kevin Gimpel with
the corresponding GitHub repo: https://github.com/hendrycks/GELUs
"""

import math

import torch
import torch.nn as nn

def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.SiLU(x.float()).type_as(x)
