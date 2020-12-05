# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import sys
import unittest

import torch

from fairseq import distributed_utils as dist_utils

from .utils import objects_are_equal, spawn_and_init


class TestDistributedUtils(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if sys.platform == "win32":
            raise unittest.SkipTest("NCCL doesn't support Windows, skipping test")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("distributed tests require 2+ GPUs, skipping")

    def test_broadcast_object_python(self):
        spawn_and_init(
            functools.partial(
                TestDistributedUtils._test_broadcast_object,
                "hello world",
            ),
            world_size=2,
        )

    def test_broadcast_object_tensor(self):
        spawn_and_init(
            functools.partial(
                TestDistributedUtils._test_broadcast_object,
                torch.rand(5),
            ),
            world_size=2,
        )

    def test_broadcast_object_complex(self):
        spawn_and_init(
            functools.partial(
                TestDistributedUtils._test_broadcast_object,
                {
                    "a": "1",
                    "b": [2, torch.rand(2, 3), 3],
                    "c": (torch.rand(2, 3), 4),
                    "d": {5, torch.rand(5)},
                    "e": torch.rand(5),
                    "f": torch.rand(5).int().cuda(),
                },
            ),
            world_size=2,
        )

    @staticmethod
    def _test_broadcast_object(ref_obj, rank, group):
        obj = dist_utils.broadcast_object(
            ref_obj if rank == 0 else None, src_rank=0, group=group
        )
        assert objects_are_equal(ref_obj, obj)


if __name__ == "__main__":
    unittest.main()
