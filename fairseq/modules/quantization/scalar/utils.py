#!/usr/bin/env python3

import logging
import re
from operator import attrgetter, itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from ..pq.utils import get_layers
from .modules import IntConv2d, IntLinear, IntEmbedding, ActivationQuantizer


def quantize_model_(model, p=0.2, bits=8, update_step=1000):
    """
    Docstring 
    """
    
    # quantize all layers
    quantized_layers = get_layers(model, "*")

    for layer in quantized_layers:

        # book-keeping
        is_master_process = (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0)
        verbose = verbose and is_master_process

        # get block size and centroids
        module = attrgetter(layer)(model)
        block_size = get_param(module, layer, block_sizes_config)
        n_centroids = get_param(module, layer, n_centroids_config)
        if verbose:
            logging.info(f"Quantizing layer {layer} with bits={bits} and QuantNoise={p}")

        # copy layer weights
        weight = module.weight.data.clone()
        is_bias = 'bias' in module.__dict__ and module.bias is not None
        bias = module.bias.data.clone() if is_bias else None

        # instantiate the quantized counterpart
        if isinstance(module, nn.Linear):
            out_features, in_features = map(
                lambda k: module.__dict__[k], ["out_features", "in_features"]
            )
            quantized_layer = IntLinear(
                in_features,
                out_features,
                bias=is_bias,
                p=p,
                update_step=update_step,
                bits=bits,
                method="histogram",
            )
            
        elif isinstance(module, nn.Embedding):
            num_embeddings, embedding_dim = map(
                lambda k: module.__dict__[k], ["num_embeddings", "embedding_dim"]
            )
            quantized_layer = IntEmbedding(
                num_embeddings, 
                embedding_dim,
                p=p,
                update_step=update_step,
                bits=bits,
                method="histogram",
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

            quantized_layer = IntConv2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=is_bias
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                p=p,
                update_step=update_step,
                bits=bits,
                method="histogram"
            )

        else:
            raise ValueError(f"Module {module} not yet supported for quantization")

        # copy layer weights 
        quantized_layer.weight = weight 
        if is_bias:
            quantizer_layer.bias = bias        
            
        # activation quantization 
        a_q = ActivationQuantizer(quantized_layer, p=p, bits=bits, method="histogram")
                
        # replace layer by its quantized counterpart
        attrsetter(layer)(model, quantized_layer)

    # return name of quantized layers
    return quantized_layers