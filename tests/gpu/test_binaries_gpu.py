# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import tempfile
import unittest
from io import StringIO

import torch
from fairseq import options
from fairseq_cli import train
from tests.utils import (
    create_dummy_data,
    generate_main,
    preprocess_lm_data,
    preprocess_translation_data,
    train_translation_model,
)


class TestTranslationGPU(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fp16") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, "fconv_iwslt_de_en", ["--fp16"])
                generate_main(data_dir)

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_memory_efficient_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_memory_efficient_fp16") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir, "fconv_iwslt_de_en", ["--memory-efficient-fp16"]
                )
                generate_main(data_dir)

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_transformer_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "transformer_iwslt_de_en",
                    [
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "64",
                        "--decoder-embed-dim",
                        "64",
                        "--fp16",
                    ],
                    run_validation=True,
                )
                generate_main(data_dir)

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_levenshtein_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_levenshtein_transformer"
            ) as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, ["--joined-dictionary"])
                train_translation_model(
                    data_dir,
                    "levenshtein_transformer",
                    [
                        "--apply-bert-init",
                        "--early-exit",
                        "6,6,6",
                        "--criterion",
                        "nat_loss",
                    ],
                    task="translation_lev",
                )
                generate_main(
                    data_dir,
                    [
                        "--task",
                        "translation_lev",
                        "--iter-decode-max-iter",
                        "9",
                        "--iter-decode-eos-penalty",
                        "0",
                        "--print-step",
                    ],
                )


def _quantize_language_model(data_dir, arch, extra_flags=None, run_validation=False):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            "language_modeling",
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--criterion",
            "adaptive_loss",
            "--adaptive-softmax-cutoff",
            "5,10,15",
            "--max-tokens",
            "500",
            "--tokens-per-sample",
            "500",
            "--save-dir",
            data_dir,
            "--max-epoch",
            "1",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
        ]
        + (extra_flags or []),
    )
    train.main(train_args)

    # try scalar quantization
    scalar_quant_train_parser = options.get_training_parser()
    scalar_quant_train_args = options.parse_args_and_arch(
        scalar_quant_train_parser,
        [
            "--task",
            "language_modeling",
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--criterion",
            "adaptive_loss",
            "--adaptive-softmax-cutoff",
            "5,10,15",
            "--max-tokens",
            "500",
            "--tokens-per-sample",
            "500",
            "--save-dir",
            data_dir,
            "--max-update",
            "3",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
            "--quant-noise-scalar",
            "0.5",
        ]
        + (extra_flags or []),
    )
    train.main(scalar_quant_train_args)

    # try iterative PQ quantization
    quantize_parser = options.get_training_parser()
    quantize_args = options.parse_args_and_arch(
        quantize_parser,
        [
            "--task",
            "language_modeling",
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--criterion",
            "adaptive_loss",
            "--adaptive-softmax-cutoff",
            "5,10,15",
            "--max-tokens",
            "50",
            "--tokens-per-sample",
            "50",
            "--max-update",
            "6",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
            "--restore-file",
            os.path.join(data_dir, "checkpoint_last.pt"),
            "--reset-optimizer",
            "--quantization-config-path",
            os.path.join(
                os.path.dirname(__file__), "transformer_quantization_config.yaml"
            ),
        ]
        + (extra_flags or []),
    )
    train.main(quantize_args)


class TestQuantization(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_quantization(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_quantization") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                # tests both scalar and iterative PQ quantization
                _quantize_language_model(data_dir, "transformer_lm")


class TestOptimizersGPU(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_flat_grads(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_flat_grads") as data_dir:
                # Use just a bit of data and tiny model to keep this test runtime reasonable
                create_dummy_data(data_dir, num_examples=10, maxlen=5)
                preprocess_translation_data(data_dir)
                with self.assertRaises(RuntimeError):
                    # adafactor isn't compatible with flat grads, which
                    # are used by default with --fp16
                    train_translation_model(
                        data_dir,
                        "lstm",
                        [
                            "--required-batch-size-multiple",
                            "1",
                            "--encoder-layers",
                            "1",
                            "--encoder-hidden-size",
                            "32",
                            "--decoder-layers",
                            "1",
                            "--optimizer",
                            "adafactor",
                            "--fp16",
                        ],
                    )
                # but it should pass once we set --fp16-no-flatten-grads
                train_translation_model(
                    data_dir,
                    "lstm",
                    [
                        "--required-batch-size-multiple",
                        "1",
                        "--encoder-layers",
                        "1",
                        "--encoder-hidden-size",
                        "32",
                        "--decoder-layers",
                        "1",
                        "--optimizer",
                        "adafactor",
                        "--fp16",
                        "--fp16-no-flatten-grads",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
