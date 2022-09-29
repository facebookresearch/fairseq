# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import logging
from collections.abc import Iterable
from itertools import repeat
from typing import List, Optional, Tuple

import torch
from torch import Tensor


# ------------------------------------------------------------------------------
#   assert_equal()
# ------------------------------------------------------------------------------


def assert_equal(value1, value2, name1=None, name2=None):
    """Asserts two values are equal otherwise raise an error."""

    str_name1 = "" if name1 is None else "{} ".format(name1)
    str_name2 = "" if name2 is None else "{} ".format(name2)
    if value1 != value2:
        str_value1 = "{}" if name1 is None else "({})"
        str_value1 = str_value1.format(value1)
        str_value2 = "{}" if name2 is None else "({})"
        str_value2 = str_value2.format(value2)
        raise ValueError(
            "Expected {}{} == {}{}".format(str_name1, str_value1, str_name2, str_value2)
        )


def fill_config(config, key, value):
    if value is not None:
        if key not in config or config[key] is None:
            config[key] = value
        assert_equal(value, config[key], "value", f'config["{key}"]')


# ------------------------------------------------------------------------------
#   check_and_return_expected()
# ------------------------------------------------------------------------------


def check_and_return_expected(value, undefined_value, expected_value, name=None):
    """
    Return the expected value while checking if the given value is undefined or
    equal to the expected value.
    """
    if (undefined_value is None and value is None) or (undefined_value == value):
        return expected_value
    if value != expected_value:
        str_name = "" if name is None else "{} ".format(name)
        str_value = "{}" if name is None else "({})"
        str_value = str_value.format(value)
        raise ValueError(
            "Expected {}{} == {}".format(str_name, str_value, expected_value)
        )
    return expected_value


# ------------------------------------------------------------------------------
#   get_time_axis()
# ------------------------------------------------------------------------------


def get_time_axis(layout):
    """
    Extract the time axis from the layout, for example for breaking sequence into
    segments.
    """
    if layout in ["TB", "TBD"]:
        return 0
    if layout in ["BT", "BTD"]:
        return 1
    if layout in ["BCTD"]:
        return 2
    raise ValueError("Unsupported layout = {}".format(layout))


# ------------------------------------------------------------------------------
#   get_batch_axis()
# ------------------------------------------------------------------------------


def get_batch_axis(layout):
    """
    Extract the batch axis from the layout
    """
    if layout in ["TB", "TBD"]:
        return 1
    if layout in ["BT", "BTD", "BCTD"]:
        return 0
    raise ValueError("Unsupported layout = {}".format(layout))


# ------------------------------------------------------------------------------
#   monotonically_increasing_and_bounded()
# ------------------------------------------------------------------------------


def monotonically_increasing_and_bounded(iterable, min=None, max=None):
    """
    Check if the elements in the given iterable are monotonically increasing and
    bounded by upper/lower bounds.
    """
    if not isinstance(iterable, Iterable):
        raise TypeError(
            "Expected iterable to be of type Iterable, got ({})".format(
                iterable.__class__.__name__
            )
        )
    for i in range(len(iterable)):
        if min is not None and iterable[i] < min:
            return False
        if max is not None and iterable[i] > max:
            return False
        if i > 0 and iterable[i] <= iterable[i - 1]:
            return False
    return True


# ------------------------------------------------------------------------------
#   to_pair()
# ------------------------------------------------------------------------------


def to_pair(value, name):
    """Make a pair (of type tuple) of given value."""
    if isinstance(value, Iterable):
        if len(value) != 2:
            raise ValueError(
                "Expected `{}` to have exactly 2 elements, got: ({})".format(
                    name, value
                )
            )
        return value
    return tuple(repeat(value, 2))


# ------------------------------------------------------------------------------
#   infer_conv_output_attrs()
# ------------------------------------------------------------------------------


# TODO(cfyeh): figure out if we can get `output_dim` without calling the module.
def infer_conv_output_attrs(
    module, input_channels, input_dim, batch_size=1, max_length=8
):
    """Get output attributes of a module with input."""
    input = torch.randn(batch_size, input_channels, max_length, input_dim)
    output = module(input)
    output_channels = output.shape[1]
    output_dim = output.shape[-1]
    return output_channels, output_dim


# ------------------------------------------------------------------------------
#   NoOp
# ------------------------------------------------------------------------------


class NoOp(torch.nn.Module):
    """
    NoOp simply passes the input as the output.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


# ------------------------------------------------------------------------------
#   Permute: a torch.nn.Module applies permutation on the input tensor.
# ------------------------------------------------------------------------------


class Permute(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, input: Tensor) -> Tensor:
        return input.permute(self.dims).contiguous()


# ------------------------------------------------------------------------------
#   lengths_to_padding_mask()
# ------------------------------------------------------------------------------


def lengths_to_padding_mask(lengths: Tensor) -> Tensor:
    """Convert lengths of shape (B, ) to padding mask."""
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(  # [0, ..., T-1]
        max_length, device=lengths.device, dtype=lengths.dtype
    ).expand(batch_size, max_length) >= lengths.unsqueeze(1)

    return padding_mask


# ------------------------------------------------------------------------------
#   lengths_to_attention_mask()
# ------------------------------------------------------------------------------


def lengths_to_attention_mask(
    lengths: Tensor,
    left_context: Optional[int] = None,
    right_context: Optional[int] = None,
) -> Optional[Tensor]:
    """
    Generate attention mask based on (lengths, left_context, right_context).
    left_context is None means unlimited left context.
    right_context is None means unlimited right context.
    """

    if left_context is None and right_context is None:
        return None

    max_length = int(torch.max(lengths).item())

    # For example, with `max_length` == 5,
    # indices = tensor([
    #     [ 0,  1,  2,  3,  4,  5],
    #     [-1,  0,  1,  2,  3,  4],
    #     [-2, -1,  0,  1,  2,  3],
    #     [-3, -2, -1,  0,  1,  2],
    #     [-4, -3, -2, -1,  0,  1],
    #     [-5, -4, -3, -2, -1,  0],
    # ])

    # In some cases the second torch.arange is created on cpu which causes a
    # failure. Adding the device option to guard against it.
    indices = torch.arange(
        max_length, device=lengths.device, dtype=lengths.dtype
    ).expand(max_length, max_length) - torch.arange(
        max_length, device=lengths.device
    ).view(
        max_length, -1
    )

    # For example, with `max_length` == 5,
    # bool_mask = tensor([
    #     [True, True, True, True, True],
    #     [True, True, True, True, True],
    #     [True, True, True, True, True],
    #     [True, True, True, True, True],
    #     [True, True, True, True, True],
    # ])
    bool_mask = (
        torch.tensor([True]).to(device=lengths.device).expand(max_length, max_length)
    )

    # For example, with `max_length` == 5, left_context == 2
    # left_mask = tensor([
    #     [ True,  True, True, True, True],
    #     [ True,  True, True, True, True],
    #     [ True,  True, True, True, True],
    #     [False,  True, True, True, True],
    #     [False, False, True, True, True],
    # ])
    if left_context is not None:
        left_mask = indices >= -left_context
        bool_mask = bool_mask & left_mask

    # For example, with `max_length` == 5, right_context == 1
    # right_mask = tensor([
    #     [True, True, False, False, False],
    #     [True, True,  True, False, False],
    #     [True, True,  True,  True, False],
    #     [True, True,  True,  True,  True],
    #     [True, True,  True,  True,  True],
    # ])
    if right_context is not None:
        right_mask = indices <= right_context
        bool_mask = bool_mask & right_mask

    bool_mask = (~bool_mask).to(device=lengths.device)
    return bool_mask


# ------------------------------------------------------------------------------
#   infer_output_norm()
# ------------------------------------------------------------------------------


def infer_output_norm(module, output_norm=None):
    """
    Infer the output norm (string and module) needed on the module gvien desired
    output normalization.
    """
    if output_norm == module.output_norm():
        # output_norm already matches module.output_norm().
        return (None, NoOp())

    if output_norm is None and module.output_norm() is not None:
        logger = logging.getLogger("infer_output_norm()")
        logger.warning(
            "trying to set output_norm ({}) ".format(output_norm)
            + "but got module.output_norm() ({}), ".format(module.output_norm())
            + "the combined output_norm() will be ({})".format(module.output_norm())
        )
        return (None, NoOp())

    if output_norm == "log_softmax":
        if module.output_norm() is not None:
            raise ValueError(
                "incompatible output_norm ({}) ".format(output_norm)
                + "and module.output_norm() ({})".format(module.output_norm())
            )
        else:
            return ("log_softmax", torch.nn.LogSoftmax(dim=-1))

    if output_norm == "softmax":
        if module.output_norm() is not None:
            raise ValueError(
                "incompatible output_norm ({}) ".format(output_norm)
                + "and module.output_norm() ({})".format(module.output_norm())
            )
        else:
            return ("softmax", torch.nn.Softmax(dim=-1))

    raise ValueError(
        "output_norm ({}) not in ".format(output_norm)
        + "supported list = [None, softmax, log_softmax]"
    )


# ------------------------------------------------------------------------------
#   infer_channels_from_layout()
# ------------------------------------------------------------------------------


def infer_channels_from_layout(layout, channels):
    """Extract the number of channels from the layout."""
    if layout in ("TBD", "BTD"):
        if channels is not None and channels != 1:
            raise ValueError(
                "Expected channels ({}) to be 1 for layout = {}".format(
                    channels, layout
                )
            )
        if channels is None:
            return 1
    return channels


# ------------------------------------------------------------------------------
#   pad_sequence()
# ------------------------------------------------------------------------------


@torch.jit.export
def pad_sequence(
    sequence: Tensor,
    time_axis: int,
    extra_left_context: int = 0,
    extra_right_context: int = 0,
) -> Tensor:
    """Pad extra left/right contexts to the sequence."""

    if extra_left_context == 0 and extra_right_context == 0:
        return sequence

    tensors_to_concat = []

    if extra_left_context:
        size = (extra_left_context,)
        fill_value = 0
        indices = torch.full(
            size=size,
            fill_value=fill_value,
            dtype=torch.long,
            device=sequence.device,
        )
        left_padding = torch.index_select(sequence, time_axis, indices)
        tensors_to_concat.append(left_padding)

    tensors_to_concat.append(sequence)

    # NOTE(cfyeh): for efficiency reason we pad 0 instead of the last frame for
    #              extra right contexts.
    if extra_right_context:
        size = list(sequence.shape)
        size[time_axis] = extra_right_context
        right_padding = torch.zeros(size, dtype=sequence.dtype, device=sequence.device)
        tensors_to_concat.append(right_padding)

    padded_sequence = torch.cat(tensors_to_concat, dim=time_axis)
    return padded_sequence


# ------------------------------------------------------------------------------
#   sequence_to_segments()
# ------------------------------------------------------------------------------


@torch.jit.export
def sequence_to_segments(
    sequence: Tensor,
    time_axis: int,
    lengths: Tensor,
    segment_size: Optional[int] = None,
    extra_left_context: int = 0,
    extra_right_context: int = 0,
) -> List[Tuple[Tensor, Tensor]]:
    """Breaks sequence into segments."""

    sequence = pad_sequence(
        sequence=sequence,
        time_axis=time_axis,
        extra_left_context=extra_left_context,
        extra_right_context=extra_right_context,
    )

    lengths = lengths + extra_left_context + extra_right_context

    segments: List[Tuple[Tensor, Tensor]] = []

    if segment_size is None:
        segments.append((sequence, lengths))
        return segments

    offset = 0
    end = sequence.shape[time_axis]
    step = segment_size
    size = extra_left_context + segment_size + extra_right_context

    while offset + extra_left_context + extra_right_context < end:
        clamped_size = min(size, end - offset)
        segment_lengths = torch.clamp(lengths - offset, min=0, max=clamped_size)
        indices = torch.arange(
            start=offset,
            end=(offset + clamped_size),
            step=1,
            dtype=torch.long,
            device=sequence.device,
        )
        segment_tensor = torch.index_select(sequence, time_axis, indices)
        segments.append((segment_tensor, segment_lengths))
        offset = offset + step

    return segments


# ------------------------------------------------------------------------------
#   segments_to_sequence()
# ------------------------------------------------------------------------------


@torch.jit.export
def segments_to_sequence(
    segments: List[Tuple[Tensor, Tensor]], time_axis: int
) -> Tuple[Tensor, Tensor]:
    """Concatenate segments into a full sequence."""
    if len(segments) == 1:
        return segments[0]

    tensors_to_concat: List[Tensor] = []
    lengths_to_stack: List[Tensor] = []

    for tensor, lengths in segments:
        tensors_to_concat.append(tensor)
        lengths_to_stack.append(lengths)

    sequence = torch.cat(tensors_to_concat, dim=time_axis)
    lengths = torch.stack(lengths_to_stack, dim=0)
    lengths = torch.sum(lengths, dim=0)

    return sequence, lengths


def lengths_to_encoder_padding_mask(lengths, batch_first: bool = False):
    """
    convert lengths (a 1-D Long/Int tensor) to 2-D binary tensor

    Args:
        lengths: a (B, )-shaped tensor
        batch_first: whether to return a (B, T) tensor

    Return:
        max_length: maximum length of B sequences
        encoder_padding_mask: a (max_length, B) binary mask, where
        [t, b] = False for t < lengths[b] and True otherwise

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
    ) > lengths.view(  # expand to (B, T)-shaped tensor
        bsz, 1
    ).expand(
        -1, max_lengths
    )
    if not batch_first:
        return encoder_padding_mask.t(), max_lengths
    else:
        return encoder_padding_mask, max_lengths


# ------------------------------------------------------------------------------
#   attention suppression
# ------------------------------------------------------------------------------


def attention_suppression(attention_weights: Tensor, scale: float):
    # B, H, qlen, klen -> B, H, qlen, 1
    attention_prob = torch.nn.functional.softmax(attention_weights.float(), dim=-1)
    attention_nozeros = attention_prob.to(torch.bool)
    nozeros_sum = torch.sum(attention_nozeros.to(torch.float), dim=-1, keepdim=True)

    # For very sparse situation, we need get round about 0s
    key_sum = torch.sum(attention_prob, dim=-1, keepdim=True)

    # nozeros_sum should > 1
    key_mean = key_sum / (nozeros_sum + 1e-8)

    # std calculation
    dis = (attention_prob - key_mean) * (attention_prob - key_mean)

    # if attention_prob[i] < threshold, then dis_masked[i] = 0; for all i
    dis_masked = torch.where(
        attention_nozeros, dis, attention_prob.new_zeros(attention_prob.size())
    )

    key_var = torch.sum(dis_masked, dim=-1, keepdim=True)
    key_var = key_var / (nozeros_sum - 1.0 + 1e-8)
    key_std = torch.sqrt(key_var)
    key_thread = key_mean - scale * key_std

    # if attention_prob[i] >= key_thread, then attention_prob[i]
    # , otherwise "-inf"
    inf_tensor = attention_prob.new_zeros(attention_prob.size()).detach()
    inf_tensor[:] = float("-inf")
    attention_weights_float = torch.where(
        attention_prob < key_thread,
        inf_tensor,
        attention_weights.float(),
    )

    return attention_weights_float.type_as(attention_weights)


def layer_norm_backward_hook(module, grad_input, grad_output, clamp_value):
    return tuple(torch.clamp(v, min=-clamp_value, max=clamp_value) for v in grad_input)
