# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from io import StringIO
import json
import os
import tempfile
import unittest

import torch

from . import test_binaries


class TestReproducibility(unittest.TestCase):

    def _test_reproducibility(self, name, extra_flags=None):
        if extra_flags is None:
            extra_flags = []

        with tempfile.TemporaryDirectory(name) as data_dir:
            with contextlib.redirect_stdout(StringIO()):
                test_binaries.create_dummy_data(data_dir)
                test_binaries.preprocess_translation_data(data_dir)

            # train epochs 1 and 2 together
            stdout = StringIO()
            with contextlib.redirect_stdout(stdout):
                test_binaries.train_translation_model(
                    data_dir, 'fconv_iwslt_de_en', [
                        '--dropout', '0.0',
                        '--log-format', 'json',
                        '--log-interval', '1',
                        '--max-epoch', '3',
                    ] + extra_flags,
                )
            stdout = stdout.getvalue()
            train_log, valid_log = map(json.loads, stdout.split('\n')[-5:-3])

            # train epoch 2, resuming from previous checkpoint 1
            os.rename(
                os.path.join(data_dir, 'checkpoint1.pt'),
                os.path.join(data_dir, 'checkpoint_last.pt'),
            )
            stdout = StringIO()
            with contextlib.redirect_stdout(stdout):
                test_binaries.train_translation_model(
                    data_dir, 'fconv_iwslt_de_en', [
                        '--dropout', '0.0',
                        '--log-format', 'json',
                        '--log-interval', '1',
                        '--max-epoch', '3',
                    ] + extra_flags,
                )
            stdout = stdout.getvalue()
            train_res_log, valid_res_log = map(json.loads, stdout.split('\n')[-5:-3])

            def cast(s):
                return round(float(s), 3)

            for k in ['train_loss', 'train_ppl', 'train_num_updates', 'train_gnorm']:
                self.assertEqual(cast(train_log[k]), cast(train_res_log[k]))
            for k in ['valid_loss', 'valid_ppl', 'valid_num_updates', 'valid_best_loss']:
                self.assertEqual(cast(valid_log[k]), cast(valid_res_log[k]))

    def test_reproducibility(self):
        self._test_reproducibility('test_reproducibility')

    @unittest.skipIf(not torch.cuda.is_available(), 'test requires a GPU')
    def test_reproducibility_fp16(self):
        self._test_reproducibility('test_reproducibility_fp16', [
            '--fp16',
            '--fp16-init-scale', '4096',
        ])

    @unittest.skipIf(not torch.cuda.is_available(), 'test requires a GPU')
    def test_reproducibility_memory_efficient_fp16(self):
        self._test_reproducibility('test_reproducibility_memory_efficient_fp16', [
            '--memory-efficient-fp16',
            '--fp16-init-scale', '4096',
        ])


if __name__ == '__main__':
    unittest.main()
