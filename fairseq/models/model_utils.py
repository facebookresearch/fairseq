# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from torch import Tensor


@torch.jit.script
def script_skip_tensor_list(x: List[Tensor], mask):
    res = [xi[mask] if xi.size(0) == mask.size(0) else xi[:, mask] for xi in x]
    outputs = []
    for i, t in enumerate(res):
        if t.numel() != 0:
            outputs.append(t)
        else:
            outputs.append(x[i])
    return outputs


@torch.jit.script
def script_skip_tensor(x: Tensor, mask):
    # None case
    if x.size(0) == 0:
        return x
    res = x[mask] if x.size(0) == mask.size(0) else x[:, mask]
    if res.numel() == 0:
        return x
    else:
        return res


@torch.jit.script
def expand_2d_or_3d_tensor(x, trg_dim: int, padding_idx: int):
    """
    Expand 2D/3D tensor on dim=1
    """
    if x is None:
        return None

    assert x.dim() == 2 or x.dim() == 3
    assert trg_dim >= x.size(1), (trg_dim, x.size())
    if trg_dim == x.size(1):
        return x

    dims = [x.size(0), trg_dim - x.size(1)]
    if x.dim() == 3:
        dims.append(x.size(2))
    x = torch.cat([x, torch.zeros(dims).to(x).fill_(padding_idx)], 1)

    return x


@torch.jit.script
def coalesce(x: Optional[Tensor], y: Tensor) -> Tensor:
    return x if x is not None else y


@torch.jit.script
def fill_tensors(
    x: Optional[Tensor], mask, y: Optional[Tensor], padding_idx: int
) -> Optional[Tensor]:
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None or x.size()[0] == 0 or y is None:
        return x
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))

    n_selected = mask.sum()
    if n_selected == 0:
        return x
    assert n_selected == y.size(0)
    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        x = expand_2d_or_3d_tensor(x, y.size(1), padding_idx)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = torch.tensor(padding_idx).type_as(x)
        if x.dim() == 2:
            x[mask, : y.size(1)] = y
        else:
            x[mask, : y.size(1), :] = y
    else:
        x[mask] = y
    return x
