# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def emulate_int(w, bits, method, scale=None, zero_point=None):
    q = globals()[f"emulate_int{bits}_{method}"]
    return q(w, scale=scale, zero_point=zero_point)


def quantize(w, scale, zero_point):
    return (torch.clamp(torch.round(w / scale + zero_point), 0, 255) - zero_point) * scale


def emulate_int8_histogram(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.HistogramObserver()
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point


def emulate_int8_channel(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
        _ = obs(w)
        scale, zero_point, ch_axis = obs.get_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point


def emulate_int8_tensor(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.MinMaxObserver()
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point
