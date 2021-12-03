#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from argparse import Namespace
from tempfile import NamedTemporaryFile
from pathlib import Path

import torch
try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None

from fairseq.data.dictionary import Dictionary
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel, s2t_transformer_s
)


@unittest.skipIf(not huggingface_hub, "Requires huggingface_hub install")
class TestHuggingFaceHub(unittest.TestCase):
    @torch.no_grad()
    def test_hf_s2t_transformer_s(self):
        hf_model_id = "facebook/s2t-small-covost2-en-de-st"
        tgt_dict = Dictionary()
        for i in range(181 - 4):
            tgt_dict.add_symbol(str(i))
        with NamedTemporaryFile(mode="w") as f:
            f.write("vocab_filename: dict.txt")
            f.flush()
            task_args = Namespace(
                data=Path(f.name).parent.as_posix(),
                config_yaml=Path(f.name).name,
            )
            task = SpeechToTextTask(task_args, tgt_dict)
            model_args = Namespace(
                input_channels=1, input_feat_per_channel=80,
                max_source_positions=6_000
            )
            s2t_transformer_s(model_args)
            model = S2TTransformerModel.build_model(model_args, task)
        incompatible_keys = model.load_from_hf_hub(pretrained_path=hf_model_id)
        self.assertTrue(len(incompatible_keys.missing_keys) == 0)
        self.assertTrue(len(incompatible_keys.unexpected_keys) == 0)


if __name__ == "__main__":
    unittest.main()
