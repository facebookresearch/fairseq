# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from operator import attrgetter, itemgetter

import numpy as np
import torch.nn as nn
import torch.distributed as dist

from .modules import PQConv2d, PQLinear, PQEmbedding
from .pq import PQ


def quantize_model_(
    model,
    size_tracker,
    layers_to_quantize,
    block_sizes_config,
    n_centroids_config,
    step=0,
    n_iter=15,
    eps=1e-6,
    max_tentatives=100,
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
              'Conv2d': ('kernel_size', {'(3, 3)': 9, '(1, 1)': 4}),
              'Linear': ('in_features', {'*': 8})
          }
          For instance, all conv2d layers with kernel size 3x3 have
          a block size of 9 and all Linear layers are quantized with
          a block size of 8, irrespective of their size.
        - n_centroids_config: dict like
          {
              'Conv2d': ('kernel_size', {'*': 256}),
              'Linear': ('in_features', {'*': 256})
          }
          For instance, all conv2d layers are quantized with 256 centroids
        - step: the layers to quantize inplace corresponding
          to layers_to_quantize[step]
    """

    quantized_layers = get_layers(model, layers_to_quantize[step])

    for layer in quantized_layers:

        # book-keeping
        is_master_process = (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0)
        verbose = verbose and is_master_process

        # get block size and centroids
        module = attrgetter(layer)(model)
        block_size = get_param(module, layer, block_sizes_config)
        n_centroids = get_param(module, layer, n_centroids_config)
        if verbose:
            logging.info(f"Quantizing layer {layer} with block size {block_size} and {n_centroids} centroids")

        # quantize layer
        weight = module.weight.data.clone()
        is_bias = 'bias' in [x[0] for x in module.named_parameters()]
        bias = module.bias.data.clone() if is_bias else None
        quantizer = PQ(
            weight,
            block_size,
            n_centroids=n_centroids,
            n_iter=n_iter,
            eps=eps,
            max_tentatives=max_tentatives,
            verbose=verbose,
        )

        # quantization performed on all GPUs with same seed
        quantizer.encode()
        centroids = quantizer.centroids.contiguous()
        assignments = quantizer.assignments.contiguous()

        # broadcast results to make sure weights are up-to-date
        if dist.is_initialized():
            dist.broadcast(centroids, 0)
            dist.broadcast(assignments, 0)

        # instantiate the quantized counterpart
        if isinstance(module, nn.Linear):
            out_features, in_features = map(
                lambda k: module.__dict__[k], ["out_features", "in_features"]
            )
            quantized_module = PQLinear(
                centroids, assignments, bias, in_features, out_features
            )
        elif isinstance(module, nn.Embedding):
            num_embeddings, embedding_dim = map(
                lambda k: module.__dict__[k], ["num_embeddings", "embedding_dim"]
            )
            quantized_module = PQEmbedding(
                centroids, assignments, num_embeddings, embedding_dim
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

            quantized_module = PQConv2d(
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
        attrsetter(layer)(model, quantized_module)

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


def get_param(module, layer_name, param_config):
    """
    Given a quantization configuration, get the right parameter
    for the module to be quantized.

    Args:
        - module: a nn.Module
        - layer_name: the name of the layer
        - param_config: a dict like
          {
              'Conv2d': ('kernel_size', {'(3, 3)': 9, '(1, 1)': 4}),
              'Linear': ('in_features', {'*': 8})
          }
          For instance, all conv2d layers with kernel size 3x3 have
          a block size of 9 and all Linear layers are quantized with
          a block size of 8, irrespective of their size.

    Remarks:
        - if 'fuzzy_name' is passed as a parameter, layers whose layer_name
          include 'fuzzy_name' will be assigned the given parameter.
          In the following example, conv.expand layers will have a block
          size of 9 while conv.reduce will have a block size of 4 and all
          other layers will have a block size of 2.
          {
              'Conv2d': ('fuzzy_name', {'expand': 9, 'reduce': 4, '*': 2}),
              'Linear': ('fuzzy_name', {'classifier': 8, 'projection': 4})
          }

    """

    layer_type = module.__class__.__name__

    if layer_type not in param_config:
        raise KeyError(f"Layer type {layer_type} not in config for layer {module}")

    feature, params = param_config[module.__class__.__name__]

    if feature != "fuzzy_name":
        feature_value = str(getattr(module, feature))
        if feature_value not in params:
            if "*" in params:
                feature_value = "*"
            else:
                raise KeyError(
                    f"{feature}={feature_value} not in config for layer {module}"
                )
    else:
        feature_values = [name for name in params if name in layer_name]
        if len(feature_values) == 0:
            if "*" in params:
                feature_value = "*"
            else:
                raise KeyError(
                    f"name={layer_name} not in config for {module}"
                )
        else:
            feature_value = feature_values[0]

    return params[feature_value]


class SizeTracker(object):
    """
    Class to keep track of the compressed network size with iPQ.

    Args:
        - model: a nn.Module

    Remarks:
        - The compressed size is the sum of three components
          for each layer in the network:
              (1) Storing the centroids given by iPQ in fp16
              (2) Storing the assignments of the blocks in int8
              (3) Storing all non-compressed elements such as biases
        - This cost in only valid if we use 256 centroids (then
          indexing can indeed by done with int8).
    """

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
