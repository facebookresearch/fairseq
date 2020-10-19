# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from operator import attrgetter

import torch.distributed as dist
import torch.nn as nn

from ..pq.utils import attrsetter, get_layers
from .modules import ActivationQuantizer, IntConv2d, IntEmbedding, IntLinear


MAPPING = {nn.Linear: IntLinear, nn.Embedding: IntEmbedding, nn.Conv2d: IntConv2d}


def quantize_model_(model, p=0.2, bits=8, update_step=3000):
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
        is_master_process = (not dist.is_initialized()) or (
            dist.is_initialized() and dist.get_rank() == 0
        )

        # recover module
        module = attrgetter(layer)(model)
        if is_master_process:
            logging.info(
                f"Quantizing layer {layer} with bits={bits} and QuantNoise={p}"
            )

        # quantization params
        q_params = {
            "p": p,
            "update_step": update_step,
            "bits": bits,
            "method": "histogram",
            "counter": 0,
        }

        # instantiate the quantized counterpart
        if isinstance(module, tuple(MAPPING.keys())):
            QuantizedModule = MAPPING[module.__class__]
            quantized_module = QuantizedModule.__new__(QuantizedModule)
            params = module.__dict__
            params.update(q_params)
            quantized_module.__dict__.update(params)

        else:
            if is_master_process:
                logging.info(f"Module {module} not yet supported for quantization")
            continue

        # activation quantization
        a_q = ActivationQuantizer(quantized_module, p=0, bits=bits, method="histogram")

        # replace layer by its quantized counterpart
        attrsetter(layer)(model, quantized_module)

    # return name of quantized layers
    return quantized_layers
