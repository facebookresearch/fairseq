#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from operator import attrgetter, itemgetter

import numpy as np
import torch
import torch.nn as nn

from .modules import QuantizedLinear
from .pq import PQ


def quantize_model_(
    model,
    size_tracker,
    layers_to_quantize,
    block_sizes_config,
    n_centroids_config,
    step=0,
    n_iter=20,
    eps=1e-6,
    max_tentatives=30,
    verbose=True,
):
    """
    Quantize a model in-place by stages. All the targeted
    layers are replaced by their quantized counterpart,
    and the model is ready for the finetuning of the
    centroids in a standard training loop (no modifications
    required). Note that we do not quantize biases.

    Args:
        - model: a nn.Module
        - size_tracker: useful for tracking quatization statistics
        - layers_to_quantize: a list containing regexps for
          filtering the layers to quantize at each stage according
          to their name (as in model.named_parameters())
        - block_sizes_config: dict like
          {
              'Conv2d': ('kernel_size', {'(3, 3)':9, '(1, 1)':4}),
              'Linear': ('in_features', {'*':8})
          }
          For instance, all conv2d layers with kernel size 3x3 have
          a block size of 9 and all Linear layers are quantized with
          a block size of 8, irrespective of their size.
        - n_centroids_config: dict like
          {
              'Conv2d': ('kernel_size', {'*':256}),
              'Linear': ('in_features', {'*':256})
          }
          For instance, all conv2d layers are quantized with 256 centroids
        - step: the layers to quantize inplace corresponding
          to layers_to_quantize[step]
    """

    quantized_layers = get_layers(model, layers_to_quantize[step])

    for layer in quantized_layers:

        # book-keeping
        if verbose:
            print(f"Quantizing layer {layer}")

        # get block size and centroids
        module = attrgetter(layer)(model)
        block_size = get_param(module, block_sizes_config)
        n_centroids = get_param(module, n_centroids_config)

        # remove spectral norm if any
        try:
            torch.nn.utils.remove_spectral_norm(module)
        except ValueError:
            pass

        # quantize layer
        weight = module.weight.clone().detach()
        bias = module.bias.clone().detach() if module.bias is not None else None
        quantizer = PQ(
            weight,
            block_size,
            n_centroids=n_centroids,
            n_iter=n_iter,
            eps=eps,
            max_tentatives=max_tentatives,
            verbose=True,
        )
        quantizer.encode()
        centroids, assignments = quantizer.centroids, quantizer.assignments

        # instantiate the quantized counterpart
        if isinstance(module, nn.Linear):
            out_features, in_features = map(
                lambda k: module.__dict__[k], ["out_features", "in_features"]
            )
            quantized_layer = QuantizedLinear(
                centroids, assignments, bias, in_features, out_features
            )
        elif isinstance(module, nn.Conv2d):
            out_channels, in_channels, kernel_size = map(
                lambda k: module.__dict__[k],
                ["out_channels", "in_channels", "kernel_size"],
            )
            stride, padding, dilation, groups, padding_mode = map(
                lambda k: module.__dict__[k],
                ["stride", "padding", "dilation", "groups", "padding_mode"],
            )
            # account for static padding in EfficientNet
            if groups > 1:
                padding = module.static_padding.padding
                padding = (padding[0], padding[1])

            quantized_layer = QuantizedConv2d(
                centroids,
                assignments,
                bias,
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
            )
        else:
            raise ValueError(f"Module {module} not yet supported for quantization")

        # replace layer by its quantized counterpart
        attrsetter(layer)(model, quantized_layer)

        # update statistics
        size_tracker.update(weight, block_size, n_centroids)

    # return name of quantized layers
    return quantized_layers


def get_layers(model, filter_regexp):
    """
    Filters out the layers according to a regexp. Note that
    we omit biases.

    Args:
        - model: a nn.Module
        - filter_regexp: a regexp to filter the layers to keep
          according to their name in model.named_parameters().
          For instance, the regexp:

             down_layers\\.[123456]\\.(conv[12]|identity\\.conv))

          is keeping blocks down_layers from 1 to 6, and inside
          each block is keeping conv1, conv2 and identity.conv.

    Remarks:
        - We add (module\\.)? at the beginning of the regexp to
          account for the possible use of nn.parallel.DataParallel
    """

    # get all parameter names
    all_layers = map(itemgetter(0), model.named_parameters())

    # remove biases
    all_layers = filter(lambda x: "bias" not in x, all_layers)

    # remove .weight in all other names (or .weight_orig is spectral norm)
    all_layers = map(lambda x: x.replace(".weight_orig", ""), all_layers)
    all_layers = map(lambda x: x.replace(".weight", ""), all_layers)

    # return filtered layers
    filter_regexp = "(module\\.)?" + "(" + filter_regexp + ")"
    r = re.compile(filter_regexp)

    return list(filter(r.match, all_layers))


def get_param(module, param_config):
    """
    Given a quantization configuration, get the right parameter
    for the module to be quantized.

    Args:
        - module: a nn.Module
        - param_config: a dict like
          {
              'Conv2d': ('kernel_size', {'(3, 3)':9, '(1, 1)':4}),
              'Linear': ('in_features', {'*':8})
          }
          For instance, all conv2d layers with kernel size 3x3 have
          a block size of 9 and all Linear layers are quantized with
          a block size of 8, irrespective of their size.
    """

    layer_type = module.__class__.__name__
    if layer_type not in param_config:
        raise KeyError(f"Layer type {layer_type} not in config for layer {module}")

    feature, params = param_config[module.__class__.__name__]
    feature_value = str(getattr(module, feature))
    if feature_value not in params:
        if "*" in params:
            feature_value = "*"
        else:
            raise KeyError(
                f"{feature}={feature_value} not in config for layer {module}"
            )

    return params[feature_value]


class SizeTracker(object):
    def __init__(self, model):
        self.model = model
        self.size_non_compressed_model = self.compute_size()
        self.size_non_quantized = self.size_non_compressed_model
        self.size_index = 0
        self.size_centroids = 0
        self.n_quantized_layers = 0

    def compute_size(self):
        """
        Computes the size of the model (in MB).
        """

        res = 0
        for _, p in self.model.named_parameters():
            res += p.numel()
        return res * 4 / 1024 / 1024

    def update(self, W, block_size, n_centroids):
        """
        Updates the running statistics when quantizing a new layer.
        """

        # bits per weights
        bits_per_weight = np.log2(n_centroids) / block_size
        self.n_quantized_layers += 1

        # size of indexing the subvectors of size block_size (in MB)
        size_index_layer = bits_per_weight * W.numel() / 8 / 1024 / 1024
        self.size_index += size_index_layer

        # size of the centroids stored in float16 (in MB)
        size_centroids_layer = n_centroids * block_size * 2 / 1024 / 1024
        self.size_centroids += size_centroids_layer

        # size of non-compressed layers, e.g. LayerNorms or biases (in MB)
        size_uncompressed_layer = W.numel() * 4 / 1024 / 1024
        self.size_non_quantized -= size_uncompressed_layer

    def __repr__(self):
        size_compressed = (
            self.size_index + self.size_centroids + self.size_non_quantized
        )
        compression_ratio = self.size_non_compressed_model / size_compressed  # NOQA
        return (
            f"Non-compressed model size: {self.size_non_compressed_model:.2f} MB. "
            f"After quantizing {self.n_quantized_layers} layers, size "
            f"(indexing + centroids + other): {self.size_index:.2f} MB + "
            f"{self.size_centroids:.2f} MB + {self.size_non_quantized:.2f} MB = "
            f"{size_compressed:.2f} MB, compression ratio: {compression_ratio:.2f}x"
        )


def attrsetter(*items):
    def resolve_attr(obj, attr):
        attrs = attr.split(".")
        head = attrs[:-1]
        tail = attrs[-1]

        for name in head:
            obj = getattr(obj, name)
        return obj, tail

    def g(obj, val):
        for attr in items:
            resolved_obj, resolved_attr = resolve_attr(obj, attr)
            setattr(resolved_obj, resolved_attr, val)

    return g
