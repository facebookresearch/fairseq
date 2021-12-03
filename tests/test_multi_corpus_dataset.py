# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import OrderedDict

import torch

from fairseq.data import LanguagePairDataset, TokenBlockDataset
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from tests.test_train import mock_dict


class TestMultiCorpusDataset(unittest.TestCase):
    def setUp(self):
        d = mock_dict()
        tokens_1 = torch.LongTensor([i for i in range(1, 5000, 2)]).view(1, -1)
        tokens_ds1 = TokenBlockDataset(
            tokens_1,
            sizes=[tokens_1.size(-1)],
            block_size=1,
            pad=0,
            eos=1,
            include_targets=False,
        )
        self.dataset_1 = LanguagePairDataset(
            tokens_ds1, tokens_ds1.sizes, d, shuffle=False
        )
        tokens_2 = torch.LongTensor([i for i in range(0, 5000, 2)]).view(1, -1)
        tokens_ds2 = TokenBlockDataset(
            tokens_2,
            sizes=[tokens_2.size(-1)],
            block_size=1,
            pad=0,
            eos=1,
            include_targets=False,
        )
        self.dataset_2 = LanguagePairDataset(
            tokens_ds2, tokens_ds2.sizes, d, shuffle=False
        )

    def _test_sample_helper(
        self,
        distribution,
    ):
        m = MultiCorpusDataset(
            OrderedDict({0: self.dataset_1, 1: self.dataset_2}),
            distribution=distribution,
            seed=0,
            sort_indices=True,
        )
        m.set_epoch(1)
        indices = m.ordered_indices()
        count_sample_from_first_dataset = 0
        items = set()
        for i in indices:
            item = m[i]["source"].item()
            if item % 2 == 1:
                count_sample_from_first_dataset += 1

            items.add(item)
        sample_from_first_ds_percentage = (
            1.0 * count_sample_from_first_dataset / len(indices)
        )
        self.assertLess(
            abs(sample_from_first_ds_percentage - distribution[0]),
            0.01,
        )
        self.assertEqual(
            len(items),
            int(
                min(len(self.dataset_1), len(indices) * distribution[0])
                + min(len(self.dataset_1), len(indices) * distribution[1])
            ),
        )
        print(distribution)

    def test_multi_corpus_dataset(self):
        for distribution in [[0.5, 0.5], [0.1, 0.9], [0.9, 0.1], [0.0, 1.0]]:
            self._test_sample_helper(distribution=distribution)
