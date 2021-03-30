# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import sys
import unittest

import torch

from fairseq.distributed import utils as dist_utils

from .utils import objects_are_equal, spawn_and_init


class DistributedTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if sys.platform == "win32":
            raise unittest.SkipTest("NCCL doesn't support Windows, skipping test")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("distributed tests require 2+ GPUs, skipping")


class TestBroadcastObject(DistributedTest):
    def test_str(self):
        spawn_and_init(
            functools.partial(
                TestBroadcastObject._test_broadcast_object, "hello world"
            ),
            world_size=2,
        )

    def test_tensor(self):
        spawn_and_init(
            functools.partial(
                TestBroadcastObject._test_broadcast_object,
                torch.rand(5),
            ),
            world_size=2,
        )

    def test_complex(self):
        spawn_and_init(
            functools.partial(
                TestBroadcastObject._test_broadcast_object,
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


class TestAllGatherList(DistributedTest):
    def test_str_equality(self):
        spawn_and_init(
            functools.partial(
                TestAllGatherList._test_all_gather_list_equality,
                "hello world",
            ),
            world_size=2,
        )

    def test_tensor_equality(self):
        spawn_and_init(
            functools.partial(
                TestAllGatherList._test_all_gather_list_equality,
                torch.rand(5),
            ),
            world_size=2,
        )

    def test_complex_equality(self):
        spawn_and_init(
            functools.partial(
                TestAllGatherList._test_all_gather_list_equality,
                {
                    "a": "1",
                    "b": [2, torch.rand(2, 3), 3],
                    "c": (torch.rand(2, 3), 4),
                    "d": {5, torch.rand(5)},
                    "e": torch.rand(5),
                    "f": torch.rand(5).int(),
                },
            ),
            world_size=2,
        )

    @staticmethod
    def _test_all_gather_list_equality(ref_obj, rank, group):
        objs = dist_utils.all_gather_list(ref_obj, group)
        for obj in objs:
            assert objects_are_equal(ref_obj, obj)

    def test_rank_tensor(self):
        spawn_and_init(
            TestAllGatherList._test_all_gather_list_rank_tensor, world_size=2
        )

    @staticmethod
    def _test_all_gather_list_rank_tensor(rank, group):
        obj = torch.tensor([rank])
        objs = dist_utils.all_gather_list(obj, group)
        for i, obj in enumerate(objs):
            assert obj.item() == i


if __name__ == "__main__":
    unittest.main()
