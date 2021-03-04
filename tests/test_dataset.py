# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from typing import Sequence

from fairseq.data import LanguagePairDataset, ListDataset, RoundRobinZipDatasets
from tests.test_train import mock_dict


def lang_pair_dataset(lengths: Sequence[int]) -> LanguagePairDataset:
    tokens = [[i] * l for i, l in enumerate(lengths)]
    return LanguagePairDataset(ListDataset(tokens), lengths, mock_dict())


def sample(id: int, length: int):
    return {"id": id, "source": [id] * length, "target": None}


class TestDataset(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_round_robin_zip_datasets(self):
        long_dataset = lang_pair_dataset([10, 9, 8, 11])
        short_dataset = lang_pair_dataset([11, 9])

        dataset = RoundRobinZipDatasets({"a": long_dataset, "b": short_dataset})
        # Dataset is now sorted by sentence length
        dataset.ordered_indices()
        assert dataset.longest_dataset is long_dataset
        self.assertEqual(dict(dataset[0]), {"a": sample(2, 8), "b": sample(1, 9)})
        # The item 2 of dataset 'a' is with item (2 % 2 = 0) of dataset 'b'
        self.assertEqual(dict(dataset[2]), {"a": sample(0, 10), "b": sample(1, 9)})

    def test_round_robin_zip_datasets_filtered(self):
        long_dataset = lang_pair_dataset([10, 20, 8, 11, 1000, 7, 12])
        short_dataset = lang_pair_dataset([11, 20, 9, 1000])

        dataset = RoundRobinZipDatasets({"a": long_dataset, "b": short_dataset})
        # Dataset is now sorted by sentence length
        idx = dataset.ordered_indices()
        idx, _ = dataset.filter_indices_by_size(idx, {"a": 19, "b": 900})
        self.assertEqual(list(idx), [0, 1, 2, 3, 4])
        self.assertEqual(dict(dataset[0]), {"a": sample(5, 7), "b": sample(2, 9)})
        self.assertEqual(dict(dataset[2]), {"a": sample(0, 10), "b": sample(1, 20)})
        self.assertEqual(dict(dataset[4]), {"a": sample(6, 12), "b": sample(0, 11)})

    def test_round_robin_zip_datasets_filtered_with_tuple(self):
        long_dataset = lang_pair_dataset([10, 20, 8, 11, 1000, 7, 12])
        short_dataset = lang_pair_dataset([11, 20, 9, 1000])

        dataset = RoundRobinZipDatasets({"a": long_dataset, "b": short_dataset})
        # Dataset is now sorted by sentence length
        idx = dataset.ordered_indices()
        idx, _ = dataset.filter_indices_by_size(idx, 19)
        self.assertEqual(list(idx), [0, 1, 2, 3, 4])
        self.assertEqual(dict(dataset[0]), {"a": sample(5, 7), "b": sample(2, 9)})
        self.assertEqual(dict(dataset[2]), {"a": sample(0, 10), "b": sample(2, 9)})
        self.assertEqual(dict(dataset[4]), {"a": sample(6, 12), "b": sample(2, 9)})
