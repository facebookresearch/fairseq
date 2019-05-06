# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import unittest
from collections import OrderedDict

import numpy as np
import torch
from fairseq.data import LanguagePairDataset, TokenBlockDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from tests.test_train import mock_dict


class TestMultiCorpusSampledDataset(unittest.TestCase):
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

    def _test_sample_helper(
        self,
        expected_sample_from_first_ds_percentage,
        num_samples=1000,
        sampling_func=None,
    ):
        # To make sure test is not flaky
        np.random.seed(0)
        if sampling_func is None:
            m = MultiCorpusSampledDataset(
                OrderedDict({0: self.dataset_1, 1: self.dataset_2}), default_key=0
            )
        else:
            m = MultiCorpusSampledDataset(
                OrderedDict({0: self.dataset_1, 1: self.dataset_2}),
                sampling_func=sampling_func,
                default_key=0,
            )
        m.ordered_indices()
        count_sample_from_first_dataset = 0
        for _ in range(num_samples):
            if m.collater([m[0], m[1]])["net_input"]["src_tokens"][0] == 1:
                count_sample_from_first_dataset += 1
        sample_from_first_ds_percentage = (
            1.0 * count_sample_from_first_dataset / num_samples
        )
        self.assertLess(
            abs(
                sample_from_first_ds_percentage
                - expected_sample_from_first_ds_percentage
            ),
            0.01,
        )

    def test_multi_corpus_sampled_dataset_uniform_sample(self):
        self._test_sample_helper(expected_sample_from_first_ds_percentage=0.5)

    def test_multi_corpus_sampled_dataset_weighted_sample(self):
        def naive_weighted_sample(weights):
            def f(l):
                v = np.random.random()
                agg = 0
                for i, weight in enumerate(weights):
                    agg += weight
                    if agg > v:
                        return i

            return f

        self._test_sample_helper(
            expected_sample_from_first_ds_percentage=0.9,
            sampling_func=naive_weighted_sample(weights=[0.9, 0.1]),
        )
