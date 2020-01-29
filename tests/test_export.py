#!/usr/bin/env python3

import argparse
import unittest

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import TransformerModel
from fairseq.modules import multihead_attention, sinusoidal_positional_embedding
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


class TestExportModels(unittest.TestCase):

    def test_export_multihead_attention(self):
        module = multihead_attention.MultiheadAttention(embed_dim=8, num_heads=2)
        torch.jit.script(module)

    def test_incremental_state_multihead_attention(self):
        module1 = multihead_attention.MultiheadAttention(embed_dim=8, num_heads=2)
        module1 = torch.jit.script(module1)
        module2 = multihead_attention.MultiheadAttention(embed_dim=8, num_heads=2)
        module2 = torch.jit.script(module2)

        state = {}
        state = module1.set_incremental_state(state, 'key', {'a': torch.tensor([1])})
        state = module2.set_incremental_state(state, 'key', {'a': torch.tensor([2])})
        v1 = module1.get_incremental_state(state, 'key')['a']
        v2 = module2.get_incremental_state(state, 'key')['a']

        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)

    def test_positional_embedding(self):
        module = sinusoidal_positional_embedding.SinusoidalPositionalEmbedding(
            embedding_dim=8, padding_idx=1
        )
        torch.jit.script(module)

    def test_export_transformer(self):
        task, parser = get_dummy_task_and_parser()
        TransformerModel.add_args(parser)
        args = parser.parse_args([])
        model = TransformerModel.build_model(args, task)
        torch.jit.script(model)
