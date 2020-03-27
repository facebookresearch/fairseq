#!/usr/bin/env python3
import math
import torch


def emulate_int(w, bits, method, scale=None, zero_point=None):
    q = globals()[f"emulate_int{bits}_{method}"]
    return q(w, scale=scale, zero_point=zero_point)


def quantize(w, scale, zero_point, factor=1):
    return (
        (
            torch.clamp(
                torch.round(w / (factor * scale) + zero_point / factor),
                0,
                256 // factor - 1,
            )
            - zero_point / factor
        )
        * factor
        * scale
    )


def emulate_int8_histogram(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.HistogramObserver()
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, factor=1), scale, zero_point


def emulate_int8_channel(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
        _ = obs(w)
        scale, zero_point, ch_axis = obs.get_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, factor=1), scale, zero_point


def emulate_int8_tensor(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.MinMaxObserver()
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, factor=1), scale, zero_point
