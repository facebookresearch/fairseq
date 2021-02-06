# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from torch.utils.checkpoint import checkpoint


class Model(nn.Module):
    def __init__(
        self, use_pytorch_checkpoint=False, use_fairseq_checkpoint=False, **kwargs
    ):
        super().__init__()
        torch.manual_seed(0)
        self.use_pytorch_checkpoint = use_pytorch_checkpoint
        self.ffn = nn.Sequential(
            nn.Linear(32, 128),
            # add a Dropout layer to test RNG save/restore
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
        )
        if use_fairseq_checkpoint:
            self.ffn = checkpoint_wrapper(self.ffn, **kwargs)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        if self.use_pytorch_checkpoint:
            x = checkpoint(self.ffn, x)
        else:
            x = self.ffn(x)
        return self.out(x)


class TestActivationCheckpointing(unittest.TestCase):
    def _test_checkpoint_wrapper(self, device, log_memory_usage=False):
        def get_loss_and_gnorm(model):
            torch.manual_seed(1)
            input = torch.rand(2, 16, 32).requires_grad_(True).to(device)
            model.zero_grad()
            loss = model(input).sum()
            loss.backward()
            gnorm = torch.norm(
                torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()])
            )
            return {"loss": loss, "gnorm": gnorm}

        model = Model().to(device)
        no_cpt = get_loss_and_gnorm(model)

        model = Model(use_pytorch_checkpoint=True).to(device)
        pyt_cpt = get_loss_and_gnorm(model)
        torch.testing.assert_allclose(no_cpt["loss"], pyt_cpt["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], pyt_cpt["gnorm"])

        model = Model(use_fairseq_checkpoint=True).to(device)
        fairseq_cpt = get_loss_and_gnorm(model)
        torch.testing.assert_allclose(no_cpt["loss"], fairseq_cpt["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], fairseq_cpt["gnorm"])

        model = Model(use_fairseq_checkpoint=True, offload_to_cpu=True).to(device)
        fairseq_cpt_offload = get_loss_and_gnorm(model)
        torch.testing.assert_allclose(no_cpt["loss"], fairseq_cpt_offload["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], fairseq_cpt_offload["gnorm"])

    def test_checkpoint_wrapper_cpu(self):
        self._test_checkpoint_wrapper(device=torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_checkpoint_wrapper_cuda(self):
        self._test_checkpoint_wrapper(device=torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
