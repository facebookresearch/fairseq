# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from torch.distributed import c10d
from torch.nn import parallel

from . import BaseFairseqModel


class DistributedFairseqModel(BaseFairseqModel):
    """
    A wrapper around a :class:`BaseFairseqModel` instance that adds support for
    distributed training.

    Anytime a method or attribute is called on this class we first try to
    forward it to the underlying DistributedDataParallel instance, otherwise we
    forward it to the original :class:`BaseFairseqModel` instance.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
    """

    def __init__(self, args, model):
        super().__init__()
        assert isinstance(model, BaseFairseqModel)
        if args.no_c10d:
            self.ddp_model = parallel.DistributedDataParallel(
                module=model,
                device_ids=[args.device_id],
                output_device=args.device_id,
                broadcast_buffers=False,
            )
        else:
            self.ddp_model = parallel._DistributedDataParallelC10d(
                module=model,
                device_ids=[args.device_id],
                output_device=args.device_id,
                broadcast_buffers=False,
                bucket_cap_mb=args.c10d_bucket_cap_mb,
            )

    def __call__(self, *args, **kwargs):
        return self.ddp_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.ddp_model.forward(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            return self.ddp_model.__getattr__(name)
        except AttributeError:
            pass
        return self.ddp_model.module.__getattr__(name)
