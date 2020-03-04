# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest

import torch

from fairseq.sequence_generator import SequenceGenerator

import tests.utils as test_utils


class TestSequenceGeneratorBase(unittest.TestCase):

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


class TestSequenceGenerator(TestSequenceGeneratorBase):

    def setUp(self):
        self.tgt_dict, self.w1, self.w2, src_tokens, src_lengths, self.model = (
            test_utils.sequence_generator_setup()
        )
        self.sample = {
            'net_input': {
                'src_tokens': src_tokens, 'src_lengths': src_lengths,
            },
        }

    def test_with_normalization(self):
        generator = SequenceGenerator(self.tgt_dict, beam_size=2)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
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
        generator = SequenceGenerator(self.tgt_dict, beam_size=2, normalize_scores=False)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
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
        generator = SequenceGenerator(self.tgt_dict, beam_size=2, len_penalty=lenpen)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
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
        generator = SequenceGenerator(self.tgt_dict, beam_size=2, len_penalty=lenpen)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
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
        generator = SequenceGenerator(self.tgt_dict, beam_size=2, max_len_b=2)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
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


class TestDiverseBeamSearch(TestSequenceGeneratorBase):

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
            self.tgt_dict, beam_size=2, diverse_beam_groups=2, diverse_beam_strength=0.,
        )
        sample = {'net_input': {'src_tokens': self.src_tokens, 'src_lengths': self.src_lengths}}
        hypos = generator.generate([self.model], sample)
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


class TestTopPSamplingSearch(TestSequenceGeneratorBase):

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
        # The minimal probability of top 2 tokens.
        self.min_top2_prob = 0.75
        # The minimal probability of the top 1 token.
        self.min_top1_prob = 0.4

        w1_prob = self.min_top1_prob
        w2_prob = self.min_top2_prob - self.min_top1_prob
        eos_prob = 1 - self.min_top2_prob

        args.beam_probs = [
            # step 0:
            torch.FloatTensor([
                # eos      w1   w2
                [0.0, unk, 1.0, 0.0],
                [0.0, unk, 1.0, 0.0],
                [0.0, unk, 1.0, 0.0],
                [0.0, unk, 1.0, 0.0],
            ]),
            # step 1:
            torch.FloatTensor([
                # eos           w1       w2
                [eos_prob, unk, w1_prob, w2_prob],
                [eos_prob, unk, w1_prob, w2_prob],
                [eos_prob, unk, w1_prob, w2_prob],
                [eos_prob, unk, w1_prob, w2_prob],
            ]),
            # step 2:
            torch.FloatTensor([
                # eos      w1   w2
                [1.0, unk, 0.0, 0.0],
                [1.0, unk, 0.0, 0.0],
                [1.0, unk, 0.0, 0.0],
                [1.0, unk, 0.0, 0.0],
            ]),
        ]

        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        self.model = task.build_model(args)
        self.tgt_dict = task.target_dictionary

    def test_topp_sampling_search_low_prob(self):
        # Given a prob low enough to top-P sampling, we expect only the top
        # 1 token to be sampled, which always results in the same output.
        low_sampling_topp = self.min_top1_prob/2.0
        generator = SequenceGenerator(
            self.tgt_dict, beam_size=2, sampling=True,
            sampling_topp=low_sampling_topp
        )
        sample = {
            'net_input': {
                'src_tokens': self.src_tokens,
                'src_lengths': self.src_lengths
            }
        }
        hypos = generator.generate([self.model], sample)
        eos, w1 = self.eos, self.w1
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [1.0, 0.4, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w1, w1, eos])
        self.assertHypoScore(hypos[0][1], [1.0, 0.4, 1.0])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w1, eos])
        self.assertHypoScore(hypos[1][0], [1.0, 0.4, 1.0])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w1, eos])
        self.assertHypoScore(hypos[1][1], [1.0, 0.4, 1.0])

    def test_topp_sampling_search_high_prob(self):
        # Given a prob high enough to top-P sampling, any of the top 2
        # tokens could be sampled. This can cause different outputs.
        high_sampling_topp = (self.min_top1_prob+self.min_top2_prob)/2.0
        generator = SequenceGenerator(
            self.tgt_dict, beam_size=2, sampling=True,
            sampling_topp=high_sampling_topp
        )
        sample = {
            'net_input': {
                'src_tokens': self.src_tokens,
                'src_lengths': self.src_lengths
            }
        }
        hypos = generator.generate([self.model], sample)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertTrue(self.hypoTokens(hypos[0][0], [w1, w1, eos]) or
                        self.hypoTokens(hypos[0][0], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[0][0], [1.0, 0.4, 1.0]) or
                        self.hypoScore(hypos[0][0], [1.0, 0.35, 1.0]))

        # sentence 1, beam 2
        self.assertTrue(self.hypoTokens(hypos[0][1], [w1, w1, eos]) or
                        self.hypoTokens(hypos[0][1], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[0][1], [1.0, 0.4, 1.0]) or
                        self.hypoScore(hypos[0][1], [1.0, 0.35, 1.0]))

        # sentence 2, beam 1
        self.assertTrue(self.hypoTokens(hypos[1][0], [w1, w1, eos]) or
                        self.hypoTokens(hypos[1][0], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[1][0], [1.0, 0.4, 1.0]) or
                        self.hypoScore(hypos[1][0], [1.0, 0.35, 1.0]))

        # sentence 2, beam 2
        self.assertTrue(self.hypoTokens(hypos[1][1], [w1, w1, eos]) or
                        self.hypoTokens(hypos[1][1], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[1][1], [1.0, 0.4, 1.0]) or
                        self.hypoScore(hypos[1][1], [1.0, 0.35, 1.0]))

    def hypoTokens(self, hypo, tokens):
        return self.tensorEqual(hypo['tokens'], torch.LongTensor(tokens))

    def hypoScore(self, hypo, pos_probs, normalized=True, lenpen=1.):
        pos_scores = torch.FloatTensor(pos_probs).log()
        if not self.almostEqual(hypo['positional_scores'], pos_scores):
            return False
        if pos_scores.numel() != hypo['tokens'].numel():
            return False
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        return abs(score - hypo['score']) < 1e-6

    def almostEqual(self, t1, t2):
        return t1.size() == t2.size() and (t1 - t2).abs().max() < 1e-4

    def tensorEqual(self, t1, t2):
        return t1.size() == t2.size() and t1.ne(t2).long().sum() == 0


if __name__ == '__main__':
    unittest.main()
