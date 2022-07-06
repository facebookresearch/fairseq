# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest

from fairseq import options
from fairseq.data.data_utils import raise_if_valid_subsets_unintentionally_ignored
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from .utils import create_dummy_data, preprocess_lm_data, train_language_model


def make_lm_config(
    data_dir=None,
    extra_flags=None,
    task="language_modeling",
    arch="transformer_lm_gpt2_tiny",
):
    task_args = [task]
    if data_dir is not None:
        task_args += [data_dir]
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            *task_args,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--max-tokens",
            "500",
            "--tokens-per-sample",
            "500",
            "--save-dir",
            data_dir,
            "--max-epoch",
            "1",
        ]
        + (extra_flags or []),
    )
    cfg = convert_namespace_to_omegaconf(train_args)
    return cfg


def write_empty_file(path):
    with open(path, "w"):
        pass
    assert os.path.exists(path)


class TestValidSubsetsErrors(unittest.TestCase):
    """Test various filesystem, clarg combinations and ensure that error raising happens as expected"""

    def _test_case(self, paths, extra_flags):
        with tempfile.TemporaryDirectory() as data_dir:
            [
                write_empty_file(os.path.join(data_dir, f"{p}.bin"))
                for p in paths + ["train"]
            ]
            cfg = make_lm_config(data_dir, extra_flags=extra_flags)
            raise_if_valid_subsets_unintentionally_ignored(cfg)

    def test_default_raises(self):
        with self.assertRaises(ValueError):
            self._test_case(["valid", "valid1"], [])
        with self.assertRaises(ValueError):
            self._test_case(
                ["valid", "valid1", "valid2"], ["--valid-subset", "valid,valid1"]
            )

    def partially_specified_valid_subsets(self):
        with self.assertRaises(ValueError):
            self._test_case(
                ["valid", "valid1", "valid2"], ["--valid-subset", "valid,valid1"]
            )
        # Fix with ignore unused
        self._test_case(
            ["valid", "valid1", "valid2"],
            ["--valid-subset", "valid,valid1", "--ignore-unused-valid-subsets"],
        )

    def test_legal_configs(self):
        self._test_case(["valid"], [])
        self._test_case(["valid", "valid1"], ["--ignore-unused-valid-subsets"])
        self._test_case(["valid", "valid1"], ["--combine-val"])
        self._test_case(["valid", "valid1"], ["--valid-subset", "valid,valid1"])
        self._test_case(["valid", "valid1"], ["--valid-subset", "valid1"])
        self._test_case(
            ["valid", "valid1"], ["--combine-val", "--ignore-unused-valid-subsets"]
        )
        self._test_case(
            ["valid1"], ["--valid-subset", "valid1"]
        )  # valid.bin doesn't need to be ignored.

    def test_disable_validation(self):
        self._test_case([], ["--disable-validation"])
        self._test_case(["valid", "valid1"], ["--disable-validation"])

    def test_dummy_task(self):
        cfg = make_lm_config(task="dummy_lm")
        raise_if_valid_subsets_unintentionally_ignored(cfg)

    def test_masked_dummy_task(self):
        cfg = make_lm_config(task="dummy_masked_lm")
        raise_if_valid_subsets_unintentionally_ignored(cfg)


class TestCombineValidSubsets(unittest.TestCase):
    def _train(self, extra_flags):
        with self.assertLogs() as logs:
            with tempfile.TemporaryDirectory("test_transformer_lm") as data_dir:
                create_dummy_data(data_dir, num_examples=64)
                preprocess_lm_data(data_dir)

                shutil.copyfile(f"{data_dir}/valid.bin", f"{data_dir}/valid1.bin")
                shutil.copyfile(f"{data_dir}/valid.idx", f"{data_dir}/valid1.idx")
                train_language_model(
                    data_dir,
                    "transformer_lm",
                    ["--max-update", "0", "--log-format", "json"] + extra_flags,
                    run_validation=False,
                )
        return [x.message for x in logs.records]

    def test_combined(self):
        flags = ["--combine-valid-subsets"]
        logs = self._train(flags)
        assert any(["valid1" in x for x in logs])  # loaded 100 examples from valid1
        assert not any(["valid1_ppl" in x for x in logs])  # metrics are combined

    def test_subsets(self):
        flags = ["--valid-subset", "valid,valid1"]
        logs = self._train(flags)
        assert any(["valid_ppl" in x for x in logs])  # loaded 100 examples from valid1
        assert any(["valid1_ppl" in x for x in logs])  # metrics are combined
