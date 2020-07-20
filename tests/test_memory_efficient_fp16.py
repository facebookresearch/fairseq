# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import unittest

import torch

from fairseq.optim.adam import FairseqAdam
from fairseq.optim.fp16_optimizer import MemoryEfficientFP16Optimizer


@unittest.skipIf(not torch.cuda.is_available(), 'test requires a GPU')
class TestMemoryEfficientFP16(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_load_state_dict(self):
        # define simple FP16 model
        model = torch.nn.Linear(5, 5).cuda().half()
        params = list(model.parameters())

        # initialize memory efficient FP16 optimizer
        optimizer = FairseqAdam(
            argparse.Namespace(
                lr=[0.00001],
                adam_betas='(0.9, 0.999)',
                adam_eps=1e-8,
                weight_decay=0.0,
            ),
            params,
        )
        me_optimizer = MemoryEfficientFP16Optimizer(
            argparse.Namespace(
                fp16_init_scale=1,
                fp16_scale_window=1,
                fp16_scale_tolerance=1,
                threshold_loss_scale=1,
                min_loss_scale=1e-4,
            ),
            params,
            optimizer,
        )

        # optimizer state is created in the first step
        loss = model(torch.rand(5).cuda().half()).sum()
        me_optimizer.backward(loss)
        me_optimizer.step()

        # reload state
        state = me_optimizer.state_dict()
        me_optimizer.load_state_dict(state)
        for k, v in me_optimizer.optimizer.state.items():
            self.assertTrue(k.dtype == torch.float16)
            for v_i in v.values():
                if torch.is_tensor(v_i):
                    self.assertTrue(v_i.dtype == torch.float32)


if __name__ == '__main__':
    unittest.main()
