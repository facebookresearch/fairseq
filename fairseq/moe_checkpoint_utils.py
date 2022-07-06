# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from collections import OrderedDict, defaultdict
from glob import glob
from typing import Dict, List

import numpy as np
import torch
from fairscale.nn.data_parallel.fsdp_optim_utils import is_singleton_tensor

from fairseq import distributed_utils
from fairseq.file_io import torch_load_cpu

OPT_KEY = "last_optimizer_state"
logger = logging.getLogger(__name__)


def merge_expert_and_shared_state(expert_state, shared_state):
    state = {}
    for key in ["cfg", "args", "extra_state", "optimizer_history"]:
        state[key] = expert_state[key]
    state["model"] = {**expert_state["model"], **shared_state["model"]}

    if OPT_KEY in expert_state:
        state[OPT_KEY] = {}
        for key in ["loss_scale", "param_groups"]:
            if key in expert_state[OPT_KEY]:
                state[OPT_KEY][key] = expert_state[OPT_KEY][key]

        if "param_id_map" in shared_state[OPT_KEY]:  # FSDP
            unflat_expert_state = _unflat_expert_tensor_state(
                expert_state[OPT_KEY], shared_state[OPT_KEY]
            )
            state[OPT_KEY]["state"] = {
                **shared_state[OPT_KEY]["state"],
                **unflat_expert_state,
            }

            state[OPT_KEY].update(
                {
                    k: v
                    for k, v in shared_state[OPT_KEY].items()
                    if k not in state[OPT_KEY]
                }
            )
        else:
            state[OPT_KEY]["state"] = {
                **expert_state[OPT_KEY]["state"],
                **shared_state[OPT_KEY]["state"],
            }
    return state


def split_shared_and_expert_states(model, optimizer):
    model_state_dict = model.state_dict()
    shared_model_state_dict = OrderedDict()
    expert_model_state_dict = OrderedDict()
    for name, value in model_state_dict.items():
        # TODO: this is a bit hacky - find a better way determine expert params
        if "expert" in name and "expert_centroids" not in name:
            expert_model_state_dict[name] = value
        else:
            shared_model_state_dict[name] = value

    shared_optimizer_state_dict = {}
    expert_optimizer_state_dict = {}
    optimizer_state_dict = optimizer.state_dict()
    for key in ["param_groups", "loss_scale"]:
        if key in optimizer_state_dict:
            expert_optimizer_state_dict[key] = optimizer_state_dict[key]
            shared_optimizer_state_dict[key] = optimizer_state_dict[key]

    param_mappings = {}
    param_id_to_is_expert = {}
    start_index = 0
    for group in optimizer.param_groups:
        # nonlocal start_index
        packed = {k: v for k, v in group.items() if k != "params"}
        for i, p in enumerate(group["params"], start_index):
            if id(p) not in param_mappings:
                param_mappings.update({id(p): i})
                param_id_to_is_expert[i] = hasattr(p, "expert") or hasattr(
                    p, "base_expert"
                )
        packed["params"] = [param_mappings[id(p)] for p in group["params"]]
        start_index += len(packed["params"])
        # return packed

    # param_groups = [pack_group(g) ]
    expert_optimizer_state_dict["state"] = {
        k: v
        for k, v in optimizer_state_dict["state"].items()
        if param_id_to_is_expert[k]
    }
    shared_optimizer_state_dict["state"] = {
        k: v
        for k, v in optimizer_state_dict["state"].items()
        if not param_id_to_is_expert[k]
    }
    return (
        (shared_model_state_dict, shared_optimizer_state_dict),
        (expert_model_state_dict, expert_optimizer_state_dict),
    )


def merge_multi_local_expert_states(expert_states: List[Dict]) -> Dict:
    merged_expert_state = {}
    for key in ["cfg", "args", "extra_state", "optimizer_history"]:
        merged_expert_state[key] = expert_states[0][key]

    if OPT_KEY in expert_states[0]:
        logger.warning(
            "Not stitching last optimizer state while merging experts. "
            "This is okay for inference but not for continued training. "
        )

    model_state_dict = {}
    for expert_group_id, expert_state in enumerate(expert_states):
        num_local_experts_in_chkpt = 1
        for key in expert_state["model"]:
            match = re.search(r"experts.([1-9][0-9]*)", key)
            if match and int(match.groups()[0]) + 1 > num_local_experts_in_chkpt:
                num_local_experts_in_chkpt = int(match.groups()[0]) + 1
        logger.info(
            f"found {num_local_experts_in_chkpt} local experts in expert_group_id={expert_group_id}"
        )
        for key, val in expert_state["model"].items():
            match = re.search(r"experts.([0-9][0-9]*)", key)
            assert (
                match is not None
            ), '"experts.([0-9][0-9]*)" pattern expected in key {key}'
            local_chkpt_expert_id = int(match.groups()[0])
            target_expert_id = (
                expert_group_id * num_local_experts_in_chkpt + local_chkpt_expert_id
            )
            key = key.replace(
                f"experts.{local_chkpt_expert_id}",
                "experts.{}".format(target_expert_id),
            )
            model_state_dict[key] = val
    merged_expert_state["model"] = model_state_dict
    return merged_expert_state


def load_expert_state(fnames):
    if len(fnames) == 1:
        return torch_load_cpu(fnames[0])
    else:
        return merge_multi_local_expert_states([torch_load_cpu(f) for f in fnames])


def assert_equal(a, b, msg=""):
    assert a == b, f"{msg}{a} != {b}"


def _unflat_expert_tensor_state(expert, shared) -> Dict:
    """called from merge_expert_and_shared_state, for FSDP only."""

    local_to_globals = defaultdict(list)
    for global_id, local_id in shared["param_id_map"].items():
        if local_id in shared["uncollected_local_ids"]:
            local_to_globals[local_id].append(global_id)

    flat_expert_state = expert["state"]
    unflat_state = {}
    for local_id, global_ids in local_to_globals.items():
        global_ids = sorted(global_ids)
        unflat_state.update({g: {} for g in global_ids})
        already_unflat = {
            k: v
            for k, v in flat_expert_state[local_id].items()
            if not torch.is_tensor(v) or is_singleton_tensor(v)
        }
        for buffer_name, flat_param in flat_expert_state[local_id].items():
            if torch.is_tensor(flat_param) and not is_singleton_tensor(flat_param):
                unflat_shapes = [
                    shared["state"][g][buffer_name].shape for g in global_ids
                ]
                numels = [np.prod(s) for s in unflat_shapes]
                unflat = zip(
                    global_ids,
                    (
                        t.view(s)
                        for (t, s) in zip(flat_param.split(numels), unflat_shapes)
                    ),
                )
                for gid, t in unflat:
                    unflat_state[gid][buffer_name] = t
                    unflat_state[gid].update(already_unflat)
    return unflat_state
