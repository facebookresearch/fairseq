#!/usr/bin/env python3

import logging
import re
from operator import attrgetter, itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from ..pq.utils import get_layers, attrsetter
from .modules import IntConv2d, IntLinear, IntEmbedding, ActivationQuantizer


def quantize_model_(model, p=0.2, bits=8, update_step=1000):
    """
    Replaces all modules with their scalar quantized counterpart and 
    registers hooks to quantize the post-ativations of those modules. 
    
    Args:
        - model: a nn.Module 
        - p: amount of noise (0 for no noise, 1 to quantize all the weights/activations)
        - bits: number of bits 
        - update_step: update quantization parameters every update_step steps 
    """
    
    # quantize all layers
    quantized_layers = get_layers(model, "(.*?)")

    for layer in quantized_layers:

        # book-keeping
        is_master_process = (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0)

        # get block size and centroids
        module = attrgetter(layer)(model)
        if is_master_process:
            logging.info(f"Quantizing layer {layer} with bits={bits} and QuantNoise={p}")

        # copy layer weights
        weight = module.weight.data.clone()
        is_bias = 'bias' in [x[0] for x in module.named_parameters()]
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
                bias=is_bias,
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
            if is_master_process:
                logging.info(f"Module {module} not yet supported for quantization")
            continue

        # copy layer weights 
        quantized_layer.weight.data = weight 
        if is_bias:
            quantized_layer.bias.data = bias        
            
        # activation quantization 
        a_q = ActivationQuantizer(quantized_layer, p=0, bits=bits, method="histogram")
                
        # replace layer by its quantized counterpart
        attrsetter(layer)(model, quantized_layer)

    # return name of quantized layers
    return quantized_layers