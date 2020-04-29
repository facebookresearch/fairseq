# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import tempfile
import unittest

import tests.utils as test_utils
import torch
from fairseq import search
from fairseq.data.dictionary import Dictionary

from fairseq.models.transformer import TransformerModel
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
from fairseq.tasks.fairseq_task import FairseqTask


DEFAULT_TEST_VOCAB_SIZE = 100


class DummyTask(FairseqTask):
    def __init__(self, args):
        super().__init__(args)
        self.dictionary = get_dummy_dictionary()
        if getattr(self.args, "ctc", False):
            self.dictionary.add_symbol("<ctc_blank>")
        self.src_dict = self.dictionary
        self.tgt_dict = self.dictionary

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.dictionary


def get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE):
    dummy_dict = Dictionary()
    # add dummy symbol to satisfy vocab size
    for id, _ in enumerate(range(vocab_size)):
        dummy_dict.add_symbol("{}".format(id), 1000)
    return dummy_dict


def get_dummy_task_and_parser():
    """
    to build a fariseq model, we need some dummy parse and task. This function
    is used to create dummy task and parser to faciliate model/criterion test

    Note: we use FbSpeechRecognitionTask as the dummy task. You may want
    to use other task by providing another function
    """
    parser = argparse.ArgumentParser(
        description="test_dummy_s2s_task", argument_default=argparse.SUPPRESS
    )
    DummyTask.add_args(parser)
    args = parser.parse_args([])
    task = DummyTask.setup_task(args)
    return task, parser


class TestJitSequenceGeneratorBase(unittest.TestCase):
    def setUp(self):
        self.task, self.parser = get_dummy_task_and_parser()
        eos = self.task.tgt_dict.eos()
        src_tokens = torch.randint(3, 50, (2, 10)).long()
        src_tokens = torch.cat((src_tokens, torch.LongTensor([[eos], [eos]])), -1)
        src_lengths = torch.LongTensor([2, 10])
        self.sample = {
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}
        }
        TransformerModel.add_args(self.parser)
        args = self.parser.parse_args([])
        args.encoder_layers = 2
        args.decoder_layers = 1
        self.transformer_model = TransformerModel.build_model(args, self.task)

    def assertOutputEqual(self, hypo, pos_probs):
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertTensorSizeEqual(hypo["positional_scores"], pos_scores)
        self.assertTensorSizeEqual(pos_scores.numel(), hypo["tokens"].numel())

    def assertTensorSizeEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)

    def assertHypoEqual(self, h1, h2):
        "Check two hypos are equal"
        self.assertTensorEqual(h1["tokens"], h2["tokens"])
        self.assertAlmostEqual(h1["positional_scores"], h2["positional_scores"])
        self.assertLess(abs(h1["score"] - h2["score"]), 1e-6)
        self.assertAlmostEqual(h1["attention"], h2["attention"])

    def _test_save_and_load(self, scripted_module):
        with tempfile.NamedTemporaryFile() as f:
            scripted_module.save(f.name)
            torch.jit.load(f.name)


class TestJitSequeneceGenerator(TestJitSequenceGeneratorBase):

    @unittest.skipIf(
        torch.__version__ < "1.5.0", "Targeting OSS scriptability for the 1.5 release"
    )
    def test_export_transformer(self):
        model = self.transformer_model
        torch.jit.script(model)

    @unittest.skipIf(
        torch.__version__ < "1.6.0", "Targeting OSS scriptability for the 1.6 release"
    )
    def test_ensemble_sequence_generator(self):
        model = self.transformer_model
        generator = SequenceGenerator(
            [model], self.task.tgt_dict, beam_size=2, no_repeat_ngram_size=2
        )
        scripted_model = torch.jit.script(generator)
        self._test_save_and_load(scripted_model)


class TestJitEnsemble(TestJitSequenceGeneratorBase):

    @unittest.skipIf(
        torch.__version__ < "1.6.0", "Targeting OSS scriptability for the 1.6 release"
    )
    def test_export_ensemble_model(self):
        model = self.transformer_model
        ensemble_models = EnsembleModel([model])
        torch.jit.script(ensemble_models)


class TestExportSearch(unittest.TestCase):
    def setUp(self):
        task, _ = get_dummy_task_and_parser()
        self.tgt_dict = task.tgt_dict
        self.min_top1_prob = 0.4

    def test_export_diverse_bs(self):
        search_strategy = search.DiverseBeamSearch(
            self.tgt_dict, num_groups=2, diversity_strength=0.0
        )
        torch.jit.script(search_strategy)

    def test_export_sampling(self):
        low_sampling_topp = self.min_top1_prob / 2.0
        search_strategy = search.Sampling(
            self.tgt_dict, sampling_topp=low_sampling_topp
        )
        torch.jit.script(search_strategy)

    def test_export_diverse_siblings_search(self):
        search_strategy = search.DiverseSiblingsSearch(
            self.tgt_dict, diversity_rate=0.5
        )
        torch.jit.script(search_strategy)


class TestSequenceGeneratorBase(unittest.TestCase):
    def assertHypoTokens(self, hypo, tokens):
        self.assertTensorEqual(hypo["tokens"], torch.LongTensor(tokens))

    def assertHypoScore(self, hypo, pos_probs, normalized=True, lenpen=1.0):
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertAlmostEqual(hypo["positional_scores"], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo["tokens"].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        self.assertLess(abs(score - hypo["score"]), 1e-6)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


class TestSequeneceGenerator(TestSequenceGeneratorBase):
    def setUp(self):
        self.tgt_dict, self.w1, self.w2, src_tokens, src_lengths, self.model = (
            test_utils.sequence_generator_setup()
        )
        self.sample = {
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}
        }

    def test_with_normalization(self):
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2)
        hypos = generator.forward(self.sample)
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
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, normalize_scores=False
        )
        hypos = generator.forward(self.sample)
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
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, len_penalty=lenpen
        )
        hypos = generator.forward(self.sample)
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
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, len_penalty=lenpen
        )
        hypos = generator.forward(self.sample)
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
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, max_len_b=2)
        hypos = generator.forward(self.sample)
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

    def test_encoder_with_different_output_len(self):
        args = self.model.encoder.args
        task = test_utils.TestTranslationTask.setup_task(args, self.tgt_dict, self.tgt_dict)
        reshaping_model = test_utils.TestReshapingModel.build_model(args, task)
        generator = SequenceGenerator([reshaping_model], self.tgt_dict, beam_size=2, max_len_b=2)
        hypos = generator.forward(self.sample)
        for sent in [0, 1]:
            for beam in [0, 1]:
                assert hypos[sent][beam]['attention'] is not None


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
        search_strategy = search.DiverseBeamSearch(self.tgt_dict, num_groups=2, diversity_strength=0.)
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy,
        )
        sample = {'net_input': {'src_tokens': self.src_tokens, 'src_lengths': self.src_lengths}}
        hypos = generator.forward(sample)
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


class TestDiverseSiblingsSearch(TestDiverseBeamSearch):
    def assertHypoScore(
        self, hypo, pos_probs, sibling_rank, diversity_rate, normalized=True, lenpen=1.0
    ):
        pos_scores = torch.FloatTensor(pos_probs).log()
        pos_scores.sub_(torch.Tensor(sibling_rank) * diversity_rate)
        self.assertAlmostEqual(hypo["positional_scores"], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo["tokens"].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        self.assertLess(abs(score - hypo["score"]), 1e-6)

    def test_diverse_beam_search(self):
        search_strategy = search.DiverseSiblingsSearch(
            self.tgt_dict, diversity_rate=0.5
        )
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy
        )
        sample = {
            "net_input": {
                "src_tokens": self.src_tokens,
                "src_lengths": self.src_lengths,
            }
        }
        hypos = generator.forward(sample)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 0.6, 1.0], [0, 1, 1], 0.5)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 0.4, 1.0], [0, 2, 1], 0.5)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.9], [0, 1, 1], 0.5)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.35, 0.9], [0, 2, 1], 0.5)


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
        search_strategy = search.Sampling(self.tgt_dict, sampling_topp=low_sampling_topp)
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        sample = {
            'net_input': {
                'src_tokens': self.src_tokens,
                'src_lengths': self.src_lengths
            }
        }
        hypos = generator.forward(sample)
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
        search_strategy = search.Sampling(self.tgt_dict, sampling_topp=high_sampling_topp)
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        sample = {
            'net_input': {
                'src_tokens': self.src_tokens,
                'src_lengths': self.src_lengths
            }
        }
        hypos = generator.forward(sample)
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


if __name__ == "__main__":
    unittest.main()
