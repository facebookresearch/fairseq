# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):

        @torch.jit.unused
        def forward(self, x):
            return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
