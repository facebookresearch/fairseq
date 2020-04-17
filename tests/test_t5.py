
import argparse
import json
import os
from pathlib import Path
import sys
import unittest
import tempfile

import torch

from fairseq.models.t5.t5_model import T5Model, t5_small_architecture
from fairseq.tasks.t5_finetuning import T5Dictionary, T5FinetuningTask


class TestT5Model(unittest.TestCase):
    """Tests for T5 model related code."""

    def test_default_vocab(self):
        dictionary = T5Dictionary()
        self.assertEqual(dictionary.pad_index, 0)
        self.assertEqual(dictionary.eos_index, 1)
        self.assertEqual(dictionary.unk_index, 2)
        self.assertEqual(len(dictionary), 32128)


class TestT5FinetuningTask(unittest.TestCase):
    """Test for T5 finetuning task."""

    def test_task_creation(self):
        vocab = T5Dictionary()
        args = argparse.Namespace()
        task = T5FinetuningTask(args, vocab, vocab)

    def test_setup_task(self):
        parser = argparse.ArgumentParser()
        T5FinetuningTask.add_args(parser)

        args = parser.parse_args(['data'])
        args.source_lang = 'in'
        args.target_lang = 'out'

        task = T5FinetuningTask.setup_task(args)

    def test_build_dataset_for_inference(self):
        parser = argparse.ArgumentParser()
        T5FinetuningTask.add_args(parser)

        args = parser.parse_args(['data'])
        args.source_lang = 'in'
        args.target_lang = 'out'

        task = T5FinetuningTask.setup_task(args)

        vocab = T5Dictionary()
        src_tokens = [
            vocab.encode_line('trivia question: name of the last part of harry potter?'),
            vocab.encode_line('trivia question: where does route 66 start on the west coast?'),
        ]
            
        src_lengths = list(map(len, src_tokens))
        dataset = task.build_dataset_for_inference(src_tokens, src_lengths)
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertEqual(item['id'], 0)
        self.assertEqual(item['source'][-1], vocab.eos_index)

    def test_build_dataset(self):
        parser = argparse.ArgumentParser()
        T5FinetuningTask.add_args(parser)

        with tempfile.TemporaryDirectory('test_t5') as data_dir:
            args = parser.parse_args([data_dir])
            args.source_lang = 'in'
            args.target_lang = 'out'
            args.dataset_impl = 'raw'

            with open(os.path.join(data_dir, 'train.in-out.in'), 'w') as h:
                h.writelines([
                    'trivia question: name of the last part of harry potter?',
                    'trivia question: where does route 66 start on the west coast?',
                ])
                h.write('\n')
            with open(os.path.join(data_dir, 'train.in-out.out'), 'w') as h:
                h.writelines([
                    'harry potter and the deathly hallows -- part 2',
                    'baker, california, usa',
                ])
                h.write('\n')

            task = T5FinetuningTask.setup_task(args)
            task.load_dataset('train')
            collated = task.datasets['train'].collater([task.datasets['train'][0]])
            prev_tokens = collated['net_input']['prev_output_tokens']
            self.assertEqual(int(prev_tokens[0][0]), T5Dictionary().pad_index)
