# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
import unittest

import torch

from fairseq.sequence_generator import SequenceGenerator

import tests.utils as test_utils


class TestSequenceGenerator(unittest.TestCase):

    def setUp(self):
        # construct dummy dictionary
        d = test_utils.dummy_dictionary(vocab_size=2)
        self.assertEqual(d.pad(), 1)
        self.assertEqual(d.eos(), 2)
        self.assertEqual(d.unk(), 3)
        self.eos = d.eos()
        self.w1 = 4
        self.w2 = 5

        # construct source data
        self.src_tokens = torch.LongTensor([
            [self.w1, self.w2, self.eos],
            [self.w1, self.w2, self.eos],
        ])
        self.src_lengths = torch.LongTensor([2, 2])
        self.encoder_input = {
            'src_tokens': self.src_tokens,
            'src_lengths': self.src_lengths,
        }

        args = argparse.Namespace()
        unk = 0.
        args.beam_probs = [
            # step 0:
            torch.FloatTensor([
                # eos      w1   w2
                # sentence 1:
                [0.0, unk, 0.9, 0.1],  # beam 1
                [0.0, unk, 0.9, 0.1],  # beam 2
                # sentence 2:
                [0.0, unk, 0.7, 0.3],
                [0.0, unk, 0.7, 0.3],
            ]),
            # step 1:
            torch.FloatTensor([
                # eos      w1   w2       prefix
                # sentence 1:
                [1.0, unk, 0.0, 0.0],  # w1: 0.9  (emit: w1 <eos>: 0.9*1.0)
                [0.0, unk, 0.9, 0.1],  # w2: 0.1
                # sentence 2:
                [0.25, unk, 0.35, 0.4],  # w1: 0.7  (don't emit: w1 <eos>: 0.7*0.25)
                [0.00, unk, 0.10, 0.9],  # w2: 0.3
            ]),
            # step 2:
            torch.FloatTensor([
                # eos      w1   w2       prefix
                # sentence 1:
                [0.0, unk, 0.1, 0.9],  # w2 w1: 0.1*0.9
                [0.6, unk, 0.2, 0.2],  # w2 w2: 0.1*0.1  (emit: w2 w2 <eos>: 0.1*0.1*0.6)
                # sentence 2:
                [0.60, unk, 0.4, 0.00],  # w1 w2: 0.7*0.4  (emit: w1 w2 <eos>: 0.7*0.4*0.6)
                [0.01, unk, 0.0, 0.99],  # w2 w2: 0.3*0.9
            ]),
            # step 3:
            torch.FloatTensor([
                # eos      w1   w2       prefix
                # sentence 1:
                [1.0, unk, 0.0, 0.0],  # w2 w1 w2: 0.1*0.9*0.9  (emit: w2 w1 w2 <eos>: 0.1*0.9*0.9*1.0)
                [1.0, unk, 0.0, 0.0],  # w2 w1 w1: 0.1*0.9*0.1  (emit: w2 w1 w1 <eos>: 0.1*0.9*0.1*1.0)
                # sentence 2:
                [0.1, unk, 0.5, 0.4],  # w2 w2 w2: 0.3*0.9*0.99  (emit: w2 w2 w2 <eos>: 0.3*0.9*0.99*0.1)
                [1.0, unk, 0.0, 0.0],  # w1 w2 w1: 0.7*0.4*0.4  (emit: w1 w2 w1 <eos>: 0.7*0.4*0.4*1.0)
            ]),
        ]

        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        self.model = task.build_model(args)
        self.tgt_dict = task.target_dictionary

    def test_with_normalization(self):
        generator = SequenceGenerator([self.model], self.tgt_dict)
        hypos = generator.generate(self.encoder_input, beam_size=2)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.4, 1.0])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.6])

    def test_without_normalization(self):
        # Sentence 1: unchanged from the normalized case
        # Sentence 2: beams swap order
        generator = SequenceGenerator([self.model], self.tgt_dict, normalize_scores=False)
        hypos = generator.generate(self.encoder_input, beam_size=2)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0], normalized=False)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0], normalized=False)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6], normalized=False)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.4, 1.0], normalized=False)

    def test_with_lenpen_favoring_short_hypos(self):
        lenpen = 0.6
        generator = SequenceGenerator([self.model], self.tgt_dict, len_penalty=lenpen)
        hypos = generator.generate(self.encoder_input, beam_size=2)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0], lenpen=lenpen)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0], lenpen=lenpen)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6], lenpen=lenpen)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.4, 1.0], lenpen=lenpen)

    def test_with_lenpen_favoring_long_hypos(self):
        lenpen = 5.0
        generator = SequenceGenerator([self.model], self.tgt_dict, len_penalty=lenpen)
        hypos = generator.generate(self.encoder_input, beam_size=2)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][0], [0.1, 0.9, 0.9, 1.0], lenpen=lenpen)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w1, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 1.0], lenpen=lenpen)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.4, 1.0], lenpen=lenpen)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.6], lenpen=lenpen)

    def test_maxlen(self):
        generator = SequenceGenerator([self.model], self.tgt_dict, maxlen=2)
        hypos = generator.generate(self.encoder_input, beam_size=2)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.1, 0.6])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w2, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.3, 0.9, 0.01])

    def test_no_stop_early(self):
        generator = SequenceGenerator([self.model], self.tgt_dict, stop_early=False)
        hypos = generator.generate(self.encoder_input, beam_size=2)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w2, w2, w2, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.3, 0.9, 0.99, 0.4, 1.0])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.4, 1.0])

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


class TestDiverseBeamSearch(unittest.TestCase):

    def setUp(self):
        # construct dummy dictionary
        d = test_utils.dummy_dictionary(vocab_size=2)
        self.assertEqual(d.pad(), 1)
        self.assertEqual(d.eos(), 2)
        self.assertEqual(d.unk(), 3)
        self.eos = d.eos()
        self.w1 = 4
        self.w2 = 5

        # construct source data
        self.src_tokens = torch.LongTensor([
            [self.w1, self.w2, self.eos],
            [self.w1, self.w2, self.eos],
        ])
        self.src_lengths = torch.LongTensor([2, 2])

        args = argparse.Namespace()
        unk = 0.
        args.beam_probs = [
            # step 0:
            torch.FloatTensor([
                # eos      w1   w2
                # sentence 1:
                [0.0, unk, 0.9, 0.1],  # beam 1
                [0.0, unk, 0.9, 0.1],  # beam 2
                # sentence 2:
                [0.0, unk, 0.7, 0.3],
                [0.0, unk, 0.7, 0.3],
            ]),
            # step 1:
            torch.FloatTensor([
                # eos      w1   w2
                # sentence 1:
                [0.0, unk, 0.6, 0.4],
                [0.0, unk, 0.6, 0.4],
                # sentence 2:
                [0.25, unk, 0.35, 0.4],
                [0.25, unk, 0.35, 0.4],
            ]),
            # step 2:
            torch.FloatTensor([
                # eos      w1   w2
                # sentence 1:
                [1.0, unk, 0.0, 0.0],
                [1.0, unk, 0.0, 0.0],
                # sentence 2:
                [0.9, unk, 0.1, 0.0],
                [0.9, unk, 0.1, 0.0],
            ]),
        ]

        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        self.model = task.build_model(args)
        self.tgt_dict = task.target_dictionary

    def test_diverse_beam_search(self):
        generator = SequenceGenerator(
            [self.model], self.tgt_dict,
            beam_size=2, diverse_beam_groups=2, diverse_beam_strength=0.,
        )
        encoder_input = {'src_tokens': self.src_tokens, 'src_lengths': self.src_lengths}
        hypos = generator.generate(encoder_input)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 0.6, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w1, w1, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 0.6, 1.0])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.9])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.9])

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
