# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch


try:
    from fairscale.optim import OSS, utils

    _has_fairscale = True
except ImportError:
    _has_fairscale = False


def shard_(optimizer, group):
    if not _has_fairscale:
        raise ImportError(
            "\n\nPlease install the fairscale package:" "\n\n  pip install fairscale"
        )

    class FairseqOSS(OSS):
        @property
        def disable_mem_eff_fp16_loading_hack(self):
            return True

        def __getattr__(self, name):
            if name.startswith("supports") and hasattr(self.optim, name):
                return getattr(self.optim, name)
            raise AttributeError(
                "'FairseqOSS' object has no attribute {0!r}".format(name)
            )

        def broadcast_global_state_dict(
            self, state_dict: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Broadcasts the relevant parts of a global state dict from rank 0 to
            all other ranks.
            """
            if self.rank == 0:

                # Create template state dict for all other keys not related to sharding
                template_state_dict = {
                    key: state_dict[key]
                    for key in state_dict
                    if key not in ("param_groups", "state")
                }
                template_state_dict["local_state_dict"] = True

                for dst_rank in range(self.world_size):
                    # Get the dst_rank's param_groups shard
                    send_state = {
                        "param_groups": state_dict["param_groups"][
                            state_dict["partition"][dst_rank][0] : state_dict[
                                "partition"
                            ][dst_rank][1]
                        ],
                        "state": state_dict["state"][dst_rank],
                    }
                    send_state.update(template_state_dict)

                    if dst_rank == 0:
                        recv_state = send_state
                    else:
                        utils.broadcast_object(
                            send_state,
                            src_rank=0,
                            group=self.group,
                            dist_device=self._device,
                        )
            else:
                empty_buffer = torch.tensor([0], dtype=torch.uint8, device=self._device)
                for dst_rank in range(1, self.world_size):
                    state = utils.broadcast_object(
                        empty_buffer,
                        src_rank=0,
                        group=self.group,
                        dist_device=self._device,
                    )
                    if dst_rank == self.rank:
                        recv_state = state

            return recv_state

    torch_optimizer = optimizer.optimizer
    optim_cls = type(torch_optimizer)

    optimizer.optimizer = FairseqOSS(
        torch_optimizer.param_groups,
        optim_cls,
        group=group,
        **optimizer.optimizer_config
    )
