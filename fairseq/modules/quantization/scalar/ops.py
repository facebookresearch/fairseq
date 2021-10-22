# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def emulate_int(w, bits, method, scale=None, zero_point=None):
    q = globals()[f"emulate_int8_{method}"]
    return q(w, scale=scale, zero_point=zero_point, bits=bits)


def quantize(w, scale, zero_point, bits=8):
    # In the default behavior, max_val = 255.
    max_val = 2 ** bits - 1
    return (
        torch.clamp(torch.round(w / scale + zero_point), 0, max_val) - zero_point
    ) * scale


def emulate_int8_histogram(w, scale=None, zero_point=None, bits=8):
    if scale is None:
        obs = torch.ao.quantization.observer.HistogramObserver()
        obs.to(device=w.device)
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, bits=bits), scale, zero_point


def emulate_int8_channel(w, scale=None, zero_point=None, bits=8):
    if scale is None:
        obs = torch.ao.quantization.observer.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
        obs.to(device=w.device)
        _ = obs(w)
        scale, zero_point, ch_axis = obs.get_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, bits=bits), scale, zero_point


def emulate_int8_tensor(w, scale=None, zero_point=None, bits=8):
    if scale is None:
        obs = torch.ao.quantization.observer.MinMaxObserver()
        obs.to(device=w.device)
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, bits=bits), scale, zero_point
