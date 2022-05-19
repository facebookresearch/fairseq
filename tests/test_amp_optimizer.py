# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import unittest

import torch
from torch.cuda.amp import GradScaler, autocast

from fairseq.optim import build_optimizer


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestGradientScalingAMP(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([2.0]).cuda().half()
        weight = 3.0
        bias = 5.0
        self.error = 1.0
        self.target = torch.tensor([self.x * weight + bias + self.error]).cuda()
        self.loss_fn = torch.nn.L1Loss()

        self.model = torch.nn.Linear(1, 1)
        self.model.weight.data = torch.tensor([[weight]])
        self.model.bias.data = torch.tensor([bias])
        self.model.cuda()
        self.params = list(self.model.parameters())

        self.namespace_dls = argparse.Namespace(
            optimizer="adam",
            lr=[0.1],
            adam_betas="(0.9, 0.999)",
            adam_eps=1e-8,
            weight_decay=0.0,
            threshold_loss_scale=1,
            min_loss_scale=1e-4,
        )
        self.scaler = GradScaler(
            init_scale=1,
            growth_interval=1,
        )

    def run_iter(self, model, params, optimizer):
        optimizer.zero_grad()
        with autocast():
            y = model(self.x)
            loss = self.loss_fn(y, self.target)
        self.scaler.scale(loss).backward()
        self.assertEqual(loss, torch.tensor(1.0, device="cuda:0", dtype=torch.float16))

        self.scaler.unscale_(optimizer)
        grad_norm = optimizer.clip_grad_norm(0)
        self.assertAlmostEqual(grad_norm.item(), 2.2361, 4)

        self.scaler.step(optimizer)
        self.scaler.update()
        self.assertEqual(
            model.weight,
            torch.tensor([[3.1]], device="cuda:0", requires_grad=True),
        )
        self.assertEqual(
            model.bias,
            torch.tensor([5.1], device="cuda:0", requires_grad=True),
        )
        self.assertEqual(self.scaler.get_scale(), 2.0)

    def test_automatic_mixed_precision(self):
        model = copy.deepcopy(self.model)
        params = list(model.parameters())
        optimizer = build_optimizer(self.namespace_dls, params)

        self.run_iter(model, params, optimizer)
