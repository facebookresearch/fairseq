#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub


@unittest.skipIf(not huggingface_hub, "Requires huggingface_hub install")
class TestHuggingFaceHub(unittest.TestCase):
    @torch.no_grad()
    def test_hf_fastspeech2(self):
        hf_model_id = "facebook/fastspeech2-en-ljspeech"
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(hf_model_id)
        self.assertTrue(len(models) > 0)


if __name__ == "__main__":
    unittest.main()
