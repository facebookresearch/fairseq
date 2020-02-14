# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import tempfile
import unittest

import tests.utils as test_utils
import torch
from fairseq.data.dictionary import Dictionary
from fairseq.fb_simple_sequence_generator import SimpleSequenceGenerator
from fairseq.models.transformer import TransformerModel
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


class TestJitSimpleSequeneceGenerator(unittest.TestCase):
    def _test_save_and_load(self, scripted_module):
        with tempfile.NamedTemporaryFile() as f:
            scripted_module.save(f.name)
            torch.jit.load(f.name)

    def test_export_transformer(self):
        task, parser = get_dummy_task_and_parser()
        TransformerModel.add_args(parser)
        args = parser.parse_args([])
        model = TransformerModel.build_model(args, task)
        torch.jit.script(model)

    @unittest.skipIf(
        torch.__version__ < "1.5.0", "Targeting OSS scriptability for the 1.5 release"
    )
    def test_simple_sequence_generator(self):
        task, parser = get_dummy_task_and_parser()
        TransformerModel.add_args(parser)
        args = parser.parse_args([])
        model = TransformerModel.build_model(args, task)
        generator = SimpleSequenceGenerator(model, task.tgt_dict, beam_size=2)
        scripted = torch.jit.script(generator)
        self._test_save_and_load(scripted)


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


class TestSimpleSequeneceGenerator(TestSequenceGeneratorBase):
    def setUp(self):
        self.tgt_dict, self.w1, self.w2, src_tokens, src_lengths, self.model = (
            test_utils.sequence_generator_setup()
        )
        self.encoder_input = {"src_tokens": src_tokens, "src_lengths": src_lengths}

    def test_with_normalization(self):
        generator = SimpleSequenceGenerator(self.model, self.tgt_dict, beam_size=2)
        hypos = generator.generate(self.encoder_input)
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
        generator = SimpleSequenceGenerator(
            self.model, self.tgt_dict, beam_size=2, normalize_scores=False
        )
        hypos = generator.forward(self.encoder_input)
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
        generator = SimpleSequenceGenerator(
            self.model, self.tgt_dict, beam_size=2, len_penalty=lenpen
        )
        hypos = generator.forward(self.encoder_input)
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
        generator = SimpleSequenceGenerator(
            self.model, self.tgt_dict, beam_size=2, len_penalty=lenpen
        )
        hypos = generator.forward(self.encoder_input)
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


if __name__ == "__main__":
    unittest.main()
