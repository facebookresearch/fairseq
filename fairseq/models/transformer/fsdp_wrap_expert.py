# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import math

from fairseq.distributed import fsdp_wrap
from fairseq.distributed import utils as dist_utils

logger = logging.getLogger(__name__)


def div_by_world_size(world_size, tensor):
    return tensor / world_size


def fsdp_wrap_expert(cfg, layer, min_num_params=0):
    # Wrap MoE layer with FSDP using a process group with all replicated ranks
    process_group = layer.moe_layer.expert_group
    world_size = dist_utils.get_data_parallel_group().size()
    pg_size = process_group.size()
    num_experts = world_size / pg_size

    for i, expert in enumerate(layer.moe_layer.experts):
        layer.moe_layer.experts[i] = fsdp_wrap(
            expert, process_group=process_group, min_num_params=0
        )
    if cfg.moe_normalize_expert_grad in {
        "sqrt_num_experts",
        "sqrt_world_size",
    }:
        # Rescale expert gradients by 1/sqrt(E), which is similar to reducing
        # the learning rate on expert parameters to adjust for smaller batch
        # size relative to dense (data parallel) parameters.
        expert_normalization_term = math.sqrt(num_experts)
    else:
        expert_normalization_term = num_experts

    for p in layer.moe_layer.experts.parameters():
        p.expert = True
        # Scale grads by world_size/pg_size so that grads match the equivalent replicated
        # world size expected within Trainer
        p.register_hook(functools.partial(div_by_world_size, expert_normalization_term))

    # Everything else gets wrapped as normal.
    layer = fsdp_wrap(layer, min_num_params=min_num_params)
    return layer
