# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
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
    train_language_model,
    train_translation_model,
)


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestMultiGPU(unittest.TestCase):
    @staticmethod
    def parse_logs(logfile):
        logs = []
        for ln in open(logfile, "r").readlines():
            try:
                logs.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return logs

    @property
    def world_size(self):
        return torch.cuda.device_count()

    def train_flags(self, mu):
        return [
            "--memory-efficient-fp16",
            "--update-freq",
            "1",
            "--seed",
            "1",
            "--log-format",
            "json",
            "--max-update",
            str(mu),
            "--tokens-per-sample",
            "20",
            "--batch-size",
            "2",
            "--share-decoder-input-output-embed",
            "--optimizer",
            "adam",
            "--max-valid-steps",
            "1",
            "--pad-to-fixed-length",
            "--sample-break-mode",
            "none",
        ]

    def _test_resume_multilingual_training(
        self, extra_clargs, arch="transformer_lm_gpt2_tiny"
    ):
        languages = ["en_XX", "fr_XX", "zh_CN"]
        save_interval = 5
        mu = 10
        flags = (
            self.train_flags(mu)
            + ["--save-interval-updates", str(save_interval), "--log-interval", "1"]
            + extra_clargs
        )
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fp16") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(
                    data_dir,
                    num_examples=int(
                        mu * 20 * self.world_size * 1.5
                    ),  # make sure enough data for max updates
                    languages=languages,
                )
                preprocess_lm_data(data_dir, languages)
                train_language_model(
                    data_dir,
                    arch,
                    flags + ["--log-file", log],
                    task="multilingual_language_modeling",
                    world_size=self.world_size,
                )
                log2 = os.path.join(data_dir, "resume.log")
                ckpt_name = f"checkpoint_1_{save_interval}.pt"
                restore_file = os.path.join(data_dir, ckpt_name)
                train_language_model(
                    data_dir,
                    arch,
                    flags
                    + ["--log-file", log2, "--restore-file", restore_file, "--no-save"],
                    task="multilingual_language_modeling",
                    world_size=self.world_size,
                )

                l1 = self.parse_logs(log)
                assert (
                    int(l1[-1]["train_num_updates"]) == mu
                ), f"The first run did not complete {mu} updates. Add more data"
                l2 = self.parse_logs(log2)

                if int(l2[0]["num_updates"]) != save_interval + 1:
                    all_ckpt_files = [
                        x for x in os.listdir(data_dir) if x.endswith(".pt")
                    ]
                    import shutil

                    shutil.move(data_dir, "last_failed_resume")
                    raise AssertionError(
                        f"Likely failed to load {ckpt_name}. {all_ckpt_files} \n LOGS: {l1} \n\n {l2}. "
                    )
                for k in [
                    "train_loss",
                    "train_num_updates",
                    "train_ppl",
                    "train_gnorm",
                ]:
                    from_scratch, resumed = float(l1[-1][k]), float(l2[-1][k])
                    # This fails without rounding!
                    assert (
                        from_scratch == resumed
                    ), f"difference at {k} {from_scratch} != {resumed}"


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestTranslationGPU(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_fp16_multigpu(self):
        self._test_multigpu("test_fp16", ["--fp16"])

    def test_slowmo_multigpu(self):
        self._test_multigpu(
            "test_slowmo", ["--ddp-backend", "slowmo", "--nprocs-per-node", "1"]
        )

    def test_slowmo_single_node_multigpu(self):
        self._test_multigpu(
            "test_slowmo_single_node",
            ["--ddp-backend", "slowmo", "--nprocs-per-node", "2"],
        )

    def _test_multigpu(self, test_name, test_args):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(test_name) as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "fconv_iwslt_de_en",
                    test_args + ["--log-file", log],
                    world_size=min(torch.cuda.device_count(), 2),
                )
                generate_main(data_dir)
                assert os.path.exists(log)

    @staticmethod
    def parse_logs(logfile):
        logs = []
        for ln in open(logfile, "r").readlines():
            try:
                logs.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return logs

    def test_resume_training_fsdp(self):
        self._test_resume_training(["--ddp-backend", "fully_sharded"])

    def test_resume_training_fsdp_sharded_state(self):
        self._test_resume_training(
            ["--ddp-backend", "fully_sharded", "--use-sharded-state"]
        )

    def test_resume_training_noc10d(self):
        self._test_resume_training([])

    def _test_resume_training(self, extra_clargs, arch="fconv_iwslt_de_en"):
        flags = [
            "--fp16",
            "--log-format",
            "json",
            "--max-update",
            "10",
            "--save-interval-updates",
            "2",
            "--log-interval",
            "1",
        ] + extra_clargs
        world_size = min(torch.cuda.device_count(), 2)
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fp16") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    arch,
                    flags + ["--log-file", log],
                    world_size=world_size,
                )
                log2 = os.path.join(data_dir, "resume.log")
                restore_file = os.path.join(data_dir, "checkpoint_1_2.pt")
                train_translation_model(
                    data_dir,
                    arch,
                    flags + ["--log-file", log2, "--restore-file", restore_file],
                    world_size=world_size,
                )

                l1 = self.parse_logs(log)
                l2 = self.parse_logs(log2)
                assert int(l2[0]["num_updates"]) == 3, f"{l1}\n\n {l2}"
                for k in [
                    "train_loss",
                    "train_num_updates",
                    "train_ppl",
                    "train_gnorm",
                ]:
                    from_scratch, resumed = l1[-1][k], l2[-1][k]
                    assert (
                        from_scratch == resumed
                    ), f"difference at {k} {from_scratch} != {resumed}"

    def test_memory_efficient_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_memory_efficient_fp16") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir, "fconv_iwslt_de_en", ["--memory-efficient-fp16"]
                )
                generate_main(data_dir)

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
    def test_amp(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_amp") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, "fconv_iwslt_de_en", ["--amp"])
                generate_main(data_dir)

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_transformer_amp(self):
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
                        "--amp",
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
                gen_config = [
                    "--task",
                    "translation_lev",
                    "--iter-decode-max-iter",
                    "9",
                    "--iter-decode-eos-penalty",
                    "0",
                    "--print-step",
                ]
                # non-ensemble generation
                generate_main(data_dir, gen_config)
                # ensemble generation
                generate_main(
                    data_dir,
                    gen_config,
                    path=os.pathsep.join(
                        [
                            os.path.join(data_dir, "checkpoint_last.pt"),
                            os.path.join(data_dir, "checkpoint_last.pt"),
                        ]
                    ),
                )

    def test_fsdp_checkpoint_generate(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fsdp_sharded") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                world_size = min(torch.cuda.device_count(), 2)
                train_translation_model(
                    data_dir,
                    "fconv_iwslt_de_en",
                    ["--log-file", log, "--ddp-backend", "fully_sharded"],
                    world_size=world_size,
                )
                generate_main(data_dir)
                assert os.path.exists(log)

    def test_fsdp_sharded_checkpoint_generate(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fsdp_sharded") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                world_size = min(torch.cuda.device_count(), 2)
                train_translation_model(
                    data_dir,
                    "fconv_iwslt_de_en",
                    [
                        "--log-file",
                        log,
                        "--ddp-backend",
                        "fully_sharded",
                        "--use-sharded-state",
                    ],
                    world_size=world_size,
                )
                generate_main(data_dir, ["--checkpoint-shard-count", str(world_size)])
                assert os.path.exists(log)


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


@unittest.skipIf(
    int(torch.__version__[2]) < 10, reason="quantized kernels are only supported on CPU"
)
@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestQuantization(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_quantization(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_quantization") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                # tests both scalar and iterative PQ quantization
                _quantize_language_model(data_dir, "transformer_lm")


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestOptimizersGPU(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

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
