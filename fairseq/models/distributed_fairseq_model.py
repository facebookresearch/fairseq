# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect

import torch.nn as nn
from fairseq.legacy_distributed_data_parallel import LegacyDistributedDataParallel


_GOSSIP_DISABLED = False
try:
    import gossip
except ImportError:
    _GOSSIP_DISABLED = True


def DistributedFairseqModel(args, model, process_group=None):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
    """
    # determine which DDP class to extend
    assert isinstance(model, nn.Module)
    if args.distributed_wrapper == "DDP" and args.ddp_backend == "c10d":
        ddp_class = nn.parallel.DistributedDataParallel
        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            bucket_cap_mb=args.bucket_cap_mb,
            process_group=process_group,
        )
        # Maintain backward compatibility
        if "check_reduction" in inspect.getargspec(ddp_class)[0]:
            init_kwargs["check_reduction"] = True
        if "find_unused_parameters" in inspect.getargspec(ddp_class)[0]:
            init_kwargs["find_unused_parameters"] = args.find_unused_parameters
    elif args.distributed_wrapper == "DDP" and args.ddp_backend == "no_c10d":
        ddp_class = LegacyDistributedDataParallel
        init_kwargs = dict(
            module=model,
            world_size=args.distributed_world_size,
            buffer_size=2 ** 28,
            process_group=process_group,
        )
    elif args.distributed_wrapper == "SlowMo":
        if _GOSSIP_DISABLED:
            raise ImportError(
                "Cannot find gossip library. Please install from: "
                "github.com/facebookresearch/stochastic_gradient_push"
            )
        ddp_class = gossip.GossipDataParallel

        # The values of slowmo_momentum below were obtained by tuning on the
        # En-De 16 dataset by training the transformer_wmt_en_de_large model
        if args.slowmo_momentum is None:
            if args.distributed_world_size <= 16:
                args.slowmo_momentum = 0.0
            elif args.distributed_world_size <= 32:
                args.slowmo_momentum = 0.2
            elif args.distributed_world_size <= 64:
                args.slowmo_momentum = 0.5
            else:
                args.slowmo_momentum = 0.6

        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            nprocs_per_node=args.nprocs_per_node,
            slowmo_momentum=args.slowmo_momentum,
            localsgd=(args.slowmo_algorithm == "LocalSGD"),
            localsgd_frequency=args.localsgd_frequency,
        )
    else:
        raise ValueError("Unknown --ddp-backend: " + args.ddp_backend)

    class _DistributedFairseqModel(ddp_class):
        """Extend DistributedDataParallel to check for missing
        attributes in the wrapped module."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__("module")
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

    return _DistributedFairseqModel(**init_kwargs)
