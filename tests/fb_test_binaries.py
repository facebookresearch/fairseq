# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from io import StringIO
import logging
import os
import tempfile
import unittest

import torch

from tests.utils import (
    create_dummy_data,
    preprocess_lm_data,
    preprocess_translation_data,
    train_translation_model,
    generate_main,
)

from tests.test_binaries import create_dummy_roberta_head_data, train_masked_lm, train_roberta_head


class TestTranslation(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @unittest.skipIf(not torch.cuda.is_available(), 'test requires a GPU')
    def test_fb_levenshtein_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_fb_levenshtein_transformer') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, ['--joined-dictionary'])
                train_translation_model(data_dir, 'fb_levenshtein_transformer', [
                    '--apply-bert-init', '--early-exit', '6,6,6',
                    '--criterion', 'nat_loss'
                ], task='translation_lev')
                generate_main(data_dir, [
                    '--task', 'translation_lev',
                    '--iter-decode-max-iter', '9',
                    '--iter-decode-eos-penalty', '0',
                    '--print-step',
                ])


class TestMaskedLanguageModel(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_linformer_roberta_masked_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_linformer_roberta_mlm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_masked_lm(data_dir, "roberta_c")

    def test_linformer_roberta_sentence_prediction(self):
        num_classes = 3
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_linformer_roberta_head") as data_dir:
                create_dummy_roberta_head_data(data_dir, num_classes=num_classes)
                preprocess_lm_data(os.path.join(data_dir, 'input0'))
                preprocess_lm_data(os.path.join(data_dir, 'label'))
                train_roberta_head(data_dir, "roberta_c", num_classes=num_classes)

    def test_linformer_roberta_regression_single(self):
        num_classes = 1
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_linformer_roberta_regression_single") as data_dir:
                create_dummy_roberta_head_data(data_dir, num_classes=num_classes, regression=True)
                preprocess_lm_data(os.path.join(data_dir, 'input0'))
                train_roberta_head(data_dir, "roberta_c", num_classes=num_classes, extra_flags=['--regression-target'])

    def test_linformer_roberta_regression_multiple(self):
        num_classes = 3
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_linformer_roberta_regression_multiple") as data_dir:
                create_dummy_roberta_head_data(data_dir, num_classes=num_classes, regression=True)
                preprocess_lm_data(os.path.join(data_dir, 'input0'))
                train_roberta_head(data_dir, "roberta_c", num_classes=num_classes, extra_flags=['--regression-target'])


if __name__ == '__main__':
    unittest.main()
