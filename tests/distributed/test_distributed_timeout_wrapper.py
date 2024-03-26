# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import signal
import time
import unittest

import torch
from torch import nn

from fairseq.distributed import DistributedTimeoutWrapper


class ModuleWithDelay(nn.Module):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def forward(self, x):
        time.sleep(self.delay)
        return x


class TestDistributedTimeoutWrapper(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_no_timeout(self):
        module = DistributedTimeoutWrapper(ModuleWithDelay(1), 0, signal.SIGINT)
        module(torch.rand(5))
        module.stop_timeout()

    def test_timeout_safe(self):
        module = DistributedTimeoutWrapper(ModuleWithDelay(1), 10, signal.SIGINT)
        module(torch.rand(5))
        module.stop_timeout()

    def test_timeout_killed(self):
        with self.assertRaises(KeyboardInterrupt):
            module = DistributedTimeoutWrapper(ModuleWithDelay(5), 1, signal.SIGINT)
            module(torch.rand(5))
            module.stop_timeout()


if __name__ == "__main__":
    unittest.main()
