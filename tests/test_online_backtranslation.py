# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Sequence

import fairseq.data.indexed_dataset as indexed_dataset
import fairseq.options
import fairseq.tasks.online_backtranslation as obt
import torch
from tests import utils


def mk_sample(tokens: Sequence[int], batch_size: int = 2) -> Dict[str, Any]:
    batch = torch.stack([torch.tensor(tokens, dtype=torch.long)] * batch_size)
    sample = {
        "net_input": {
            "src_tokens": batch,
            "prev_output_tokens": batch,
            "src_lengths": torch.tensor([len(tokens)] * batch_size, dtype=torch.long),
        },
        "target": batch[:, 1:],
    }
    return sample


def mk_dataset(num_samples: int, max_len: int, output: Path):
    output.parent.mkdir(exist_ok=True)
    idx = indexed_dataset.IndexedDatasetBuilder(str(output))
    data = torch.randint(5, 100, (num_samples, max_len))
    lengths = torch.randint(3, max_len, (num_samples,))
    for d, l in zip(data, lengths):
        d[0] = 0
        idx.add_item(d[:l])
    idx.finalize(output.with_suffix(".idx"))
    assert output.exists()
    assert output.with_suffix(".idx").exists()


class OnlineBacktranslationTest(unittest.TestCase):

    tmp_dir = Path(tempfile.mkdtemp(suffix="OnlineBacktranslationTest"))

    @classmethod
    def obt_task(
        cls, languages: Sequence[str], data: Path = None, language_mapping: str = None
    ):
        dict_path = cls.tmp_dir / "dict.txt"
        if not dict_path.exists():
            dictionary = utils.dummy_dictionary(100)
            dictionary.save(str(dict_path))

        if data is not None:
            (data / "dict.txt").write_text(dict_path.read_text())
        else:
            data = cls.tmp_dir
        assert len(languages) >= 2

        kwargs = {
            "arch": "transformer",
            # --max-sentences=1 for better predictability of batches
            "max_sentences": 1,
            # Use characteristics dimensions
            "encoder_layers": 3,
            "encoder_embed_dim": 12,
            "encoder_ffn_embed_dim": 14,
            "encoder_attention_heads": 4,
            "decoder_layers": 3,
            "decoder_embed_dim": 12,
            "decoder_output_dim": 12,
            "decoder_ffn_embed_dim": 14,
            "decoder_attention_heads": 4,
            # Disable dropout so we have comparable tests.
            "dropout": 0,
            "attention_dropout": 0,
            "activation_dropout": 0,
            "encoder_layerdrop": 0,
        }

        args = fairseq.options.get_args(
            data,
            task="online_backtranslation",
            mono_langs=",".join(languages),
            valid_lang_pairs=f"{languages[0]}-{languages[1]}",
            tokens_per_sample=256,
            language_mapping=language_mapping,
            **kwargs,
        )
        task = obt.OnlineBackTranslationTask.setup_task(args)
        # we need to build the model to have the correct dictionary
        model = task.build_model(task.args)
        return task, model

    def tmp_path(self, test_case: str) -> Path:
        return Path(tempfile.mkdtemp(test_case, dir=self.tmp_dir))

    def test_lang_tokens(self):
        task, model = self.obt_task(["en", "ro", "zh"])
        assert obt._lang_token("en") in task.dictionary
        assert obt._lang_token("ro") in task.dictionary
        assert obt._lang_token("zh") in task.dictionary

        en_bos = obt._lang_token_index(task.common_dict, "en")
        assert "en" == task.common_dict[en_bos].strip("_")
        zh_bos = obt._lang_token_index(task.common_dict, "zh")
        assert "zh" == task.common_dict[zh_bos].strip("_")
        zh_sample = mk_sample([zh_bos, 16, 14, 12, 10])

        # we expect to receive the bos token for translation
        assert task.get_bos_token_from_sample(zh_sample) == en_bos

    def test_backtranslate_sample(self):
        task, model = self.obt_task(["en", "ro", "zh"])

        en_bos = obt._lang_token_index(task.common_dict, "en")
        zh_bos = obt._lang_token_index(task.common_dict, "zh")
        sample = mk_sample([zh_bos, 16, 14, 12, 10])

        task.backtranslate_sample(sample, "zh", "en")
        target_zh = list(sample["target"][0])
        assert target_zh == [16, 14, 12, 10]  # original zh sentence
        generated_en = sample["net_input"]["src_tokens"][0]
        assert generated_en[0] == en_bos

    def test_train_dataset(self):
        data = self.tmp_path("test_train_dataset")
        mk_dataset(20, 10, data / "en" / "train.bin")
        mk_dataset(10, 10, data / "zh" / "train.bin")
        task, model = self.obt_task(["en", "zh"], data)
        task.load_dataset("train")

        en_bos = obt._lang_token_index(task.common_dict, "en")
        zh_bos = obt._lang_token_index(task.common_dict, "zh")

        train = task.datasets["train"]
        train.ordered_indices()
        train.prefetch([0, 19])
        sample_0 = train[0]
        sample_19 = train[19]
        self.assertEqual(
            set(sample_0.keys()), {"en-BT", "en-DENOISE", "zh-BT", "zh-DENOISE"}
        )
        for sample in (sample_0, sample_19):
            self.assertEqual(sample["en-BT"]["source"][0], en_bos)
            # bt target isn't ready to look at.
            self.assertEqual(sample["en-DENOISE"]["source"][0], en_bos)
            # TODO What could we check on the target side ?

        for i in range(10):
            # Zh dataset is shorter, and is wrapped around En dataset.
            train.prefetch([i, i + 10])
            self.assertEqual(
                list(train[i]["zh-DENOISE"]["source"]),
                list(train[i + 10]["zh-DENOISE"]["source"]),
            )
            self.assertEqual(train[i]["zh-DENOISE"]["source"][0].item(), zh_bos)

        # Sorted by increasing len
        self.assertLess(
            len(sample_0["en-BT"]["source"]), len(sample_19["en-BT"]["source"])
        )

    def test_valid_dataset(self):
        data = self.tmp_path("test_valid_dataset")
        mk_dataset(10, 21, data / "valid.en-zh.en.bin")
        mk_dataset(10, 21, data / "valid.en-zh.zh.bin")

        task, model = self.obt_task(["en", "zh"], data)
        valid = task.load_dataset("valid")
        en_bos = obt._lang_token_index(task.common_dict, "en")

        assert valid is not None
        valid.prefetch(range(10))
        sample_0 = valid[0]
        sample_9 = valid[9]
        self.assertEqual(sample_0["id"], 0)
        self.assertEqual(sample_9["id"], 9)
        self.assertEqual(sample_0["source"][0], en_bos)
        self.assertEqual(sample_9["source"][0], en_bos)
        # TODO: could we test the target side ?

    def assertFnMatch(self, fn, values):
        for x, y in values.items():
            fn_x = fn(x)
            self.assertEqual(fn_x, y, f"Fn has wrong value: fn({x}) = {fn_x} != {y}")

    def test_piecewise_linear_fn(self):
        self.assertFnMatch(
            obt.PiecewiseLinearFn.from_string("1.0"), {0: 1, 100: 1, 500: 1, 1000: 1}
        )
        self.assertFnMatch(
            obt.PiecewiseLinearFn.from_string("0:1,1000:0"),
            {0: 1, 500: 0.5, 1000: 0, 2000: 0},
        )
        self.assertFnMatch(
            obt.PiecewiseLinearFn.from_string("0:0,1000:1"),
            {0: 0, 500: 0.5, 1000: 1, 2000: 1},
        )
        self.assertFnMatch(
            obt.PiecewiseLinearFn.from_string("0:0,1000:1,2000:0"),
            {0: 0, 500: 0.5, 1000: 1, 1500: 0.5, 2000: 0, 3000: 0},
        )
