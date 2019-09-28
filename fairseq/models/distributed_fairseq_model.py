# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from torch.nn import parallel

from fairseq.distributed_utils import c10d_status
from fairseq.legacy_distributed_data_parallel import LegacyDistributedDataParallel

from . import BaseFairseqModel


def DistributedFairseqModel(args, model):
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
    assert isinstance(model, BaseFairseqModel)
    if args.ddp_backend == 'c10d':
        if c10d_status.is_default:
            ddp_class = parallel.DistributedDataParallel
        elif c10d_status.has_c10d:
            ddp_class = parallel._DistributedDataParallelC10d
        else:
            raise Exception(
                'Can\'t find c10d version of DistributedDataParallel. '
                'Please update PyTorch.'
            )
        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=False,
            bucket_cap_mb=args.bucket_cap_mb,
        )
    elif args.ddp_backend == 'no_c10d':
        if c10d_status.is_default:
            ddp_class = parallel.deprecated.DistributedDataParallel
        else:
            ddp_class = parallel.DistributedDataParallel
        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=False,
        )
    elif args.ddp_backend == 'legacy':
        ddp_class = LegacyDistributedDataParallel
        init_kwargs = dict (
            module=model,
            world_size=args.distributed_world_size,
            bucket_cap_mb=args.bucket_cap_mb,
        )
    else:
        raise ValueError('Unknown --ddp-backend: ' + args.ddp_backend)

    class _DistributedFairseqModel(ddp_class):
        """Extend DistributedDataParallel to check for missing
        attributes in the wrapped module."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__('module')
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

    return _DistributedFairseqModel(**init_kwargs)
