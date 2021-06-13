# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import os
import tempfile
import unittest
from io import StringIO

import torch

from . import test_binaries


class TestReproducibility(unittest.TestCase):
    def _test_reproducibility(
        self,
        name,
        extra_flags=None,
        delta=0.0001,
        resume_checkpoint="checkpoint1.pt",
        max_epoch=3,
    ):
        def get_last_log_stats_containing_string(log_records, search_string):
            for log_record in logs.records[::-1]:
                if isinstance(log_record.msg, str) and search_string in log_record.msg:
                    return json.loads(log_record.msg)

        if extra_flags is None:
            extra_flags = []

        with tempfile.TemporaryDirectory(name) as data_dir:
            with self.assertLogs() as logs:
                test_binaries.create_dummy_data(data_dir)
                test_binaries.preprocess_translation_data(data_dir)

            # train epochs 1 and 2 together
            with self.assertLogs() as logs:
                test_binaries.train_translation_model(
                    data_dir,
                    "fconv_iwslt_de_en",
                    [
                        "--dropout",
                        "0.0",
                        "--log-format",
                        "json",
                        "--log-interval",
                        "1",
                        "--max-epoch",
                        str(max_epoch),
                    ]
                    + extra_flags,
                )
            train_log = get_last_log_stats_containing_string(logs.records, "train_loss")
            valid_log = get_last_log_stats_containing_string(logs.records, "valid_loss")

            # train epoch 2, resuming from previous checkpoint 1
            os.rename(
                os.path.join(data_dir, resume_checkpoint),
                os.path.join(data_dir, "checkpoint_last.pt"),
            )
            with self.assertLogs() as logs:
                test_binaries.train_translation_model(
                    data_dir,
                    "fconv_iwslt_de_en",
                    [
                        "--dropout",
                        "0.0",
                        "--log-format",
                        "json",
                        "--log-interval",
                        "1",
                        "--max-epoch",
                        str(max_epoch),
                    ]
                    + extra_flags,
                )
            train_res_log = get_last_log_stats_containing_string(
                logs.records, "train_loss"
            )
            valid_res_log = get_last_log_stats_containing_string(
                logs.records, "valid_loss"
            )

            for k in ["train_loss", "train_ppl", "train_num_updates", "train_gnorm"]:
                self.assertAlmostEqual(
                    float(train_log[k]), float(train_res_log[k]), delta=delta
                )
            for k in [
                "valid_loss",
                "valid_ppl",
                "valid_num_updates",
                "valid_best_loss",
            ]:
                self.assertAlmostEqual(
                    float(valid_log[k]), float(valid_res_log[k]), delta=delta
                )

    def test_reproducibility(self):
        self._test_reproducibility("test_reproducibility")

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_reproducibility_fp16(self):
        self._test_reproducibility(
            "test_reproducibility_fp16",
            [
                "--fp16",
                "--fp16-init-scale",
                "4096",
            ],
            delta=0.011,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_reproducibility_memory_efficient_fp16(self):
        self._test_reproducibility(
            "test_reproducibility_memory_efficient_fp16",
            [
                "--memory-efficient-fp16",
                "--fp16-init-scale",
                "4096",
            ],
        )

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_reproducibility_amp(self):
        self._test_reproducibility(
            "test_reproducibility_amp",
            [
                "--amp",
                "--fp16-init-scale",
                "4096",
            ],
            delta=0.011,
        )

    def test_mid_epoch_reproducibility(self):
        self._test_reproducibility(
            "test_mid_epoch_reproducibility",
            ["--save-interval-updates", "3"],
            resume_checkpoint="checkpoint_1_3.pt",
            max_epoch=1,
        )


if __name__ == "__main__":
    unittest.main()
