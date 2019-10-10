# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def calc_mean_invstddev(feature):
    if len(feature.size()) != 2:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = feature.mean(0)
    var = feature.var(0)
    # avoid division by ~zero
    eps = 1e-8
    if (var < eps).any():
        return mean, 1.0 / (torch.sqrt(var) + eps)
    return mean, 1.0 / torch.sqrt(var)


def apply_mv_norm(features):
    mean, invstddev = calc_mean_invstddev(features)
    res = (features - mean) * invstddev
    return res


def lengths_to_encoder_padding_mask(lengths, batch_first=False):
    """
    convert lengths (a 1-D Long/Int tensor) to 2-D binary tensor

    Args:
        lengths: a (B, )-shaped tensor

    Return:
        max_length: maximum length of B sequences
        encoder_padding_mask: a (max_length, B) binary mask, where
        [t, b] = 0 for t < lengths[b] and 1 otherwise

    TODO:
        kernelize this function if benchmarking shows this function is slow
    """
    max_lengths = torch.max(lengths).item()
    bsz = lengths.size(0)
    encoder_padding_mask = torch.arange(
        max_lengths
    ).to(  # a (T, ) tensor with [0, ..., T-1]
        lengths.device
    ).view(  # move to the right device
        1, max_lengths
    ).expand(  # reshape to (1, T)-shaped tensor
        bsz, -1
    ) >= lengths.view(  # expand to (B, T)-shaped tensor
        bsz, 1
    ).expand(
        -1, max_lengths
    )
    if not batch_first:
        return encoder_padding_mask.t(), max_lengths
    else:
        return encoder_padding_mask, max_lengths


def encoder_padding_mask_to_lengths(
    encoder_padding_mask, max_lengths, batch_size, device
):
    """
    convert encoder_padding_mask (2-D binary tensor) to a 1-D tensor

    Conventionally, encoder output contains a encoder_padding_mask, which is
    a 2-D mask in a shape (T, B), whose (t, b) element indicate whether
    encoder_out[t, b] is a valid output (=0) or not (=1). Occasionally, we
    need to convert this mask tensor to a 1-D tensor in shape (B, ), where
    [b] denotes the valid length of b-th sequence

    Args:
        encoder_padding_mask: a (T, B)-shaped binary tensor or None; if None,
        indicating all are valid
    Return:
        seq_lengths: a (B,)-shaped tensor, where its (b, )-th element is the
        number of valid elements of b-th sequence

        max_lengths: maximum length of all sequence, if encoder_padding_mask is
        not None, max_lengths must equal to encoder_padding_mask.size(0)

        batch_size: batch size; if encoder_padding_mask is
        not None, max_lengths must equal to encoder_padding_mask.size(1)

        device: which device to put the result on
    """
    if encoder_padding_mask is None:
        return torch.Tensor([max_lengths] * batch_size).to(torch.int32).to(device)

    assert encoder_padding_mask.size(0) == max_lengths, "max_lengths does not match"
    assert encoder_padding_mask.size(1) == batch_size, "batch_size does not match"

    return max_lengths - torch.sum(encoder_padding_mask, dim=0)
