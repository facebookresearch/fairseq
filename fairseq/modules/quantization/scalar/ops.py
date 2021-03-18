# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import logging

def emulate_int(w, bits, method, scale=None, zero_point=None):
    q = globals()[f"emulate_int{bits}_{method}"]
    return q(w, scale=scale, zero_point=zero_point)

def quantize(w, scale, zero_point, size=8):
    return (
        torch.clamp(torch.round(w / scale + zero_point), 0, 2**size - 1) - zero_point
    ) * scale

def get_obs(size, observer_type):
    if observer_type == "hist":
        obs = torch.quantization.observer.HistogramObserver()
    elif observer_type == "channel":
        obs = torch.quantization.observer_type.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
    elif observer_type == "tensor":
        obs = torch.quantization.observer.MinMaxObserver()
    if size != 8:
        obs.quant_min, obs.quant_max = 0, 2**size - 1
        obs.has_customized_qrange = True
    return obs

def get_scale_zero_point(w, obs, observer_type):
    if observer_type == "hist":
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
    elif observer_type == "channel":
        _ = obs(w)
        scale, zero_point, ch_axis = obs.get_qparams()
    elif observer_type == "tensor":
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
    scale = scale.cuda().type_as(w)
    zero_point = zero_point.cuda().type_as(w)
    return scale, zero_point

def emulate_func(w, scale, zero_point, size, observer_type):
    if scale is None:
        obs = get_obs(size, observer_type)
        scale, zero_point = get_scale_zero_point(w, obs, observer_type)
    return quantize(w, scale, zero_point, size), scale, zero_point

def emulate_int1_histogram(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=1, observer_type="hist")

def emulate_int1_channel(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=1, observer_type="channel")

def emulate_int1_tensor(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=1, observer_type="tensor")

def emulate_int4_histogram(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=4, observer_type="hist")

def emulate_int4_channel(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=4, observer_type="channel")

def emulate_int4_tensor(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=1, observer_type="tensor")

def emulate_int8_histogram(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=8, observer_type="hist")

def emulate_int8_channel(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=1, observer_type="channel")

def emulate_int8_tensor(w, scale=None, zero_point=None):
    return emulate_func(w, scale, zero_point, size=1, observer_type="tensor")
