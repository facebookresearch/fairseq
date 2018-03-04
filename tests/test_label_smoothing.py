# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
import copy
import unittest

import torch
from torch.autograd import Variable

from fairseq import utils
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

import tests.utils as test_utils


class TestLabelSmoothing(unittest.TestCase):

    def setUp(self):
        # build dictionary
        self.d = test_utils.dummy_dictionary(3)
        vocab = len(self.d)
        self.assertEqual(vocab, 4 + 3)  # 4 special + 3 tokens
        self.assertEqual(self.d.pad(), 1)
        self.assertEqual(self.d.eos(), 2)
        self.assertEqual(self.d.unk(), 3)
        pad, eos, unk, w1, w2, w3 = 1, 2, 3, 4, 5, 6

        # build dataset
        self.data = [
            # the first batch item has padding
            {'source': torch.LongTensor([w1, eos]), 'target': torch.LongTensor([w1, eos])},
            {'source': torch.LongTensor([w1, eos]), 'target': torch.LongTensor([w1, w1, eos])},
        ]
        self.sample = next(test_utils.dummy_dataloader(self.data))

        # build model
        self.args = argparse.Namespace()
        self.args.sentence_avg = False
        self.args.probs = torch.FloatTensor([
            #      pad   eos  unk   w1   w2   w3
            [0.05, 0.05, 0.1, 0.05, 0.3, 0.4, 0.05],
            [0.05, 0.10, 0.2, 0.05, 0.2, 0.3, 0.10],
            [0.05, 0.15, 0.3, 0.05, 0.1, 0.2, 0.15],
        ]).unsqueeze(0).expand(2, 3, 7)  # add batch dimension
        self.model = test_utils.TestModel.build_model(self.args, self.d, self.d)

    def test_nll_loss(self):
        self.args.label_smoothing = 0.1
        nll_crit = CrossEntropyCriterion(self.args, self.d, self.d)
        smooth_crit = LabelSmoothedCrossEntropyCriterion(self.args, self.d, self.d)
        nll_loss, nll_sample_size, nll_logging_output = nll_crit(self.model, self.sample)
        smooth_loss, smooth_sample_size, smooth_logging_output = smooth_crit(self.model, self.sample)
        self.assertLess(abs(nll_loss - nll_logging_output['loss']), 1e-6)
        self.assertLess(abs(nll_loss - smooth_logging_output['nll_loss']), 1e-6)

    def test_padding(self):
        self.args.label_smoothing = 0.1
        crit = LabelSmoothedCrossEntropyCriterion(self.args, self.d, self.d)
        loss, _, logging_output = crit(self.model, self.sample)

        def get_one_no_padding(idx):
            # create a new sample with just a single batch item so that there's
            # no padding
            sample1 = next(test_utils.dummy_dataloader([self.data[idx]]))
            args1 = copy.copy(self.args)
            args1.probs = args1.probs[idx, :, :].unsqueeze(0)
            model1 = test_utils.TestModel.build_model(args1, self.d, self.d)
            loss1, _, _ = crit(model1, sample1)
            return loss1

        loss1 = get_one_no_padding(0)
        loss2 = get_one_no_padding(1)
        self.assertAlmostEqual(loss, loss1 + loss2)

    def test_reduction(self):
        self.args.label_smoothing = 0.1
        crit = LabelSmoothedCrossEntropyCriterion(self.args, self.d, self.d)
        loss, _, logging_output = crit(self.model, self.sample, reduce=True)
        unreduced_loss, _, _ = crit(self.model, self.sample, reduce=False)
        self.assertAlmostEqual(loss, unreduced_loss.sum())

    def test_zero_eps(self):
        self.args.label_smoothing = 0.0
        nll_crit = CrossEntropyCriterion(self.args, self.d, self.d)
        smooth_crit = LabelSmoothedCrossEntropyCriterion(self.args, self.d, self.d)
        nll_loss, nll_sample_size, nll_logging_output = nll_crit(self.model, self.sample)
        smooth_loss, smooth_sample_size, smooth_logging_output = smooth_crit(self.model, self.sample)
        self.assertAlmostEqual(nll_loss, smooth_loss)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
