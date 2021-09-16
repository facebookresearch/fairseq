# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def prob_check(tensor, eps=1e-10):
    assert not torch.isnan(tensor).any(), (
        "Nan in a probability tensor."
    )
    # Add the eps here to prevent errors introduced by precision
    assert tensor.le(1.0 + eps).all() and tensor.ge(0.0 - eps).all(), (
        "Incorrect values in a probability tensor"
        ", 0.0 <= tensor <= 1.0"
    )


def exclusive_cumprod(tensor, dim: int, eps: float = 1e-10):
    """
    Implementing exclusive cumprod.
    There is cumprod in pytorch, however there is no exclusive mode.
    cumprod(x) = [x1, x1x2, x2x3x4, ..., prod_{i=1}^n x_i]
    exclusive means
    cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
    """
    tensor_size = list(tensor.size())
    tensor_size[dim] = 1
    return_tensor = safe_cumprod(
        torch.cat([torch.ones(tensor_size).type_as(tensor), tensor], dim=dim),
        dim=dim,
        eps=eps,
    )

    if dim == 0:
        return return_tensor[:-1]
    elif dim == 1:
        return return_tensor[:, :-1]
    elif dim == 2:
        return return_tensor[:, :, :-1]
    else:
        raise RuntimeError(
            "Cumprod on dimension 3 and more is not implemented"
        )


def safe_cumprod(tensor, dim: int, eps: float = 1e-10):
    """
    An implementation of cumprod to prevent precision issue.
    cumprod(x)
    = [x1, x1x2, x1x2x3, ....]
    = [exp(log(x1)), exp(log(x1) + log(x2)), exp(log(x1) + log(x2) + log(x3)), ...]
    = exp(cumsum(log(x)))
    """

    if (tensor + eps < 0).any().item():
        raise RuntimeError(
            "Safe cumprod can only take non-negative tensors as input."
            "Consider use torch.cumprod if you want to calculate negative values."
        )

    log_tensor = torch.log(tensor + eps)
    cumsum_log_tensor = torch.cumsum(log_tensor, dim)
    exp_cumsum_log_tensor = torch.exp(cumsum_log_tensor)
    return exp_cumsum_log_tensor


def moving_sum(x, start_idx: int, end_idx: int):
    """
    From MONOTONIC CHUNKWISE ATTENTION
    https://arxiv.org/pdf/1712.05382.pdf
    Equation (18)

    x = [x_1, x_2, ..., x_N]
    MovingSum(x, start_idx, end_idx)_n = Sigma_{m=n−(start_idx−1)}^{n+end_idx-1} x_m
    for n in {1, 2, 3, ..., N}

    x : src_len, batch_size
    start_idx : start idx
    end_idx : end idx

    Example
    src_len = 5
    batch_size = 3
    x =
       [[ 0, 5, 10],
        [ 1, 6, 11],
        [ 2, 7, 12],
        [ 3, 8, 13],
        [ 4, 9, 14]]

    MovingSum(x, 3, 1) =
       [[ 0,  5, 10],
        [ 1, 11, 21],
        [ 3, 18, 33],
        [ 6, 21, 36],
        [ 9, 24, 39]]

    MovingSum(x, 1, 3) =
       [[ 3, 18, 33],
        [ 6, 21, 36],
        [ 9, 24, 39],
        [ 7, 17, 27],
        [ 4,  9, 14]]
    """
    # TODO: Make dimension configurable
    assert start_idx > 0 and end_idx > 0
    batch_size, tgt_len, src_len = x.size()
    x = x.view(-1, src_len).unsqueeze(1)
    # batch_size, 1, src_len
    moving_sum_weight = torch.ones([1, 1, end_idx + start_idx - 1]).type_as(x)

    moving_sum = torch.nn.functional.conv1d(
        x, moving_sum_weight, padding=start_idx + end_idx - 1
    ).squeeze(1)

    moving_sum = moving_sum[:, end_idx:-start_idx]

    assert src_len == moving_sum.size(1)
    assert batch_size * tgt_len == moving_sum.size(0)

    moving_sum = moving_sum.view(batch_size, tgt_len, src_len)

    return moving_sum
