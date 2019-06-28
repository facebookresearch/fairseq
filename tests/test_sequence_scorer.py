# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
import unittest

import torch

from fairseq.sequence_scorer import SequenceScorer

import tests.utils as test_utils


class TestSequenceScorer(unittest.TestCase):

    def test_sequence_scorer(self):
        # construct dummy dictionary
        d = test_utils.dummy_dictionary(vocab_size=2)
        self.assertEqual(d.pad(), 1)
        self.assertEqual(d.eos(), 2)
        self.assertEqual(d.unk(), 3)
        eos = d.eos()
        w1 = 4
        w2 = 5

        # construct dataloader
        data = [
            {
                'source': torch.LongTensor([w1, w2, eos]),
                'target': torch.LongTensor([w1, w2, w1, eos]),
            },
            {
                'source': torch.LongTensor([w2, eos]),
                'target': torch.LongTensor([w2, w1, eos]),
            },
            {
                'source': torch.LongTensor([w2, eos]),
                'target': torch.LongTensor([w2, eos]),
            },
        ]
        data_itr = test_utils.dummy_dataloader(data)

        # specify expected output probabilities
        args = argparse.Namespace()
        unk = 0.
        args.beam_probs = [
            # step 0:
            torch.FloatTensor([
                # eos      w1   w2
                [0.0, unk, 0.6, 0.4],  # sentence 1
                [0.0, unk, 0.4, 0.6],  # sentence 2
                [0.0, unk, 0.7, 0.3],  # sentence 3
            ]),
            # step 1:
            torch.FloatTensor([
                # eos      w1   w2
                [0.0, unk, 0.2, 0.7],  # sentence 1
                [0.0, unk, 0.8, 0.2],  # sentence 2
                [0.7, unk, 0.1, 0.2],  # sentence 3
            ]),
            # step 2:
            torch.FloatTensor([
                # eos       w1    w2
                [0.10, unk, 0.50, 0.4],  # sentence 1
                [0.15, unk, 0.15, 0.7],  # sentence 2
                [0.00, unk, 0.00, 0.0],  # sentence 3
            ]),
            # step 3:
            torch.FloatTensor([
                # eos      w1    w2
                [0.9, unk, 0.05, 0.05],  # sentence 1
                [0.0, unk, 0.00, 0.0],  # sentence 2
                [0.0, unk, 0.00, 0.0],  # sentence 3
            ]),
        ]
        expected_scores = [
            [0.6, 0.7, 0.5, 0.9],  # sentence 1
            [0.6, 0.8, 0.15],  # sentence 2
            [0.3, 0.7],  # sentence 3
        ]

        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        model = task.build_model(args)
        scorer = SequenceScorer(task.target_dictionary)
        for sample in data_itr:
            hypos = task.inference_step(scorer, [model], sample)
            for id, hypos_id in zip(sample['id'].tolist(), hypos):
                self.assertHypoTokens(hypos_id[0], data[id]['target'])
                self.assertHypoScore(hypos_id[0], expected_scores[id])

    def assertHypoTokens(self, hypo, tokens):
        self.assertTensorEqual(hypo['tokens'], torch.LongTensor(tokens))

    def assertHypoScore(self, hypo, pos_probs, normalized=True, lenpen=1.):
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertAlmostEqual(hypo['positional_scores'], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo['tokens'].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel()**lenpen
        self.assertLess(abs(score - hypo['score']), 1e-6)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


if __name__ == '__main__':
    unittest.main()
