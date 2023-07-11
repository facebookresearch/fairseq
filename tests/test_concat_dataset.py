# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from fairseq.data import LanguagePairDataset, TokenBlockDataset
from fairseq.data.concat_dataset import ConcatDataset
from tests.test_train import mock_dict


class TestConcatDataset(unittest.TestCase):
    def setUp(self):
        d = mock_dict()
        tokens_1 = torch.LongTensor([1]).view(1, -1)
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
        tokens_2 = torch.LongTensor([2]).view(1, -1)
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

    def test_concat_dataset_basics(self):
        d = ConcatDataset([self.dataset_1, self.dataset_2])
        assert len(d) == 2
        assert d[0]["source"][0] == 1
        assert d[1]["source"][0] == 2

        d = ConcatDataset([self.dataset_1, self.dataset_2], sample_ratios=[1, 2])
        assert len(d) == 3
        assert d[0]["source"][0] == 1
        assert d[1]["source"][0] == 2
        assert d[2]["source"][0] == 2

        d = ConcatDataset([self.dataset_1, self.dataset_2], sample_ratios=[2, 1])
        assert len(d) == 3
        assert d[0]["source"][0] == 1
        assert d[1]["source"][0] == 1
        assert d[2]["source"][0] == 2
