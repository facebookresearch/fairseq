# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
from io import StringIO
import os
import random
import sys
import tempfile
import unittest

import torch

from fairseq import options

import preprocess
import train
import generate
import interactive
import eval_lm


class TestTranslation(unittest.TestCase):

    def test_fconv(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_fconv') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'fconv_iwslt_de_en')
                generate_main(data_dir)

    def test_raw(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_fconv_raw') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, ['--output-format', 'raw'])
                train_translation_model(data_dir, 'fconv_iwslt_de_en', ['--raw-text'])
                generate_main(data_dir, ['--raw-text'])

    def test_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_fp16') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'fconv_iwslt_de_en', ['--fp16'])
                generate_main(data_dir)

    def test_memory_efficient_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_memory_efficient_fp16') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'fconv_iwslt_de_en', ['--memory-efficient-fp16'])
                generate_main(data_dir)

    def test_update_freq(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_update_freq') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'fconv_iwslt_de_en', ['--update-freq', '3'])
                generate_main(data_dir)

    def test_max_positions(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_max_positions') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                with self.assertRaises(Exception) as context:
                    train_translation_model(
                        data_dir, 'fconv_iwslt_de_en', ['--max-target-positions', '5'],
                    )
                self.assertTrue(
                    'skip this example with --skip-invalid-size-inputs-valid-test' in str(context.exception)
                )
                train_translation_model(
                    data_dir, 'fconv_iwslt_de_en',
                    ['--max-target-positions', '5', '--skip-invalid-size-inputs-valid-test'],
                )
                with self.assertRaises(Exception) as context:
                    generate_main(data_dir)
                generate_main(data_dir, ['--skip-invalid-size-inputs-valid-test'])

    def test_generation(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_sampling') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'fconv_iwslt_de_en')
                generate_main(data_dir, [
                    '--sampling',
                    '--sampling-temperature', '2',
                    '--beam', '2',
                    '--nbest', '2',
                ])
                generate_main(data_dir, [
                    '--sampling',
                    '--sampling-topk', '3',
                    '--beam', '2',
                    '--nbest', '2',
                ])
                generate_main(data_dir, ['--prefix-size', '2'])

    def test_lstm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_lstm') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'lstm_wiseman_iwslt_de_en', [
                    '--encoder-layers', '2',
                    '--decoder-layers', '2',
                ])
                generate_main(data_dir)

    def test_lstm_bidirectional(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_lstm_bidirectional') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'lstm', [
                    '--encoder-layers', '2',
                    '--encoder-bidirectional',
                    '--encoder-hidden-size', '256',
                    '--decoder-layers', '2',
                ])
                generate_main(data_dir)

    def test_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_transformer') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'transformer_iwslt_de_en')
                generate_main(data_dir)

    def test_lightconv(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_lightconv') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'lightconv_iwslt_de_en', [
                    '--encoder-conv-type', 'lightweight',
                    '--decoder-conv-type', 'lightweight',
                ])
                generate_main(data_dir)

    def test_dynamicconv(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_dynamicconv') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'lightconv_iwslt_de_en', [
                    '--encoder-conv-type', 'dynamic',
                    '--decoder-conv-type', 'dynamic',
                ])
                generate_main(data_dir)


class TestStories(unittest.TestCase):

    def test_fconv_self_att_wp(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_fconv_self_att_wp') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                config = [
                    '--encoder-layers', '[(512, 3)] * 2',
                    '--decoder-layers', '[(512, 3)] * 2',
                    '--decoder-attention', 'True',
                    '--encoder-attention', 'False',
                    '--gated-attention', 'True',
                    '--self-attention', 'True',
                    '--project-input', 'True',
                ]
                train_translation_model(data_dir, 'fconv_self_att_wp', config)
                generate_main(data_dir)

                # fusion model
                os.rename(os.path.join(data_dir, 'checkpoint_last.pt'), os.path.join(data_dir, 'pretrained.pt'))
                config.extend([
                    '--pretrained', 'True',
                    '--pretrained-checkpoint', os.path.join(data_dir, 'pretrained.pt'),
                    '--save-dir', os.path.join(data_dir, 'fusion_model'),
                ])
                train_translation_model(data_dir, 'fconv_self_att_wp', config)


class TestLanguageModeling(unittest.TestCase):

    def test_fconv_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_fconv_lm') as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_language_model(data_dir, 'fconv_lm')
                eval_lm_main(data_dir)


def create_dummy_data(data_dir, num_examples=1000, maxlen=20):

    def _create_dummy_data(filename):
        data = torch.rand(num_examples * maxlen)
        data = 97 + torch.floor(26 * data).int()
        with open(os.path.join(data_dir, filename), 'w') as h:
            offset = 0
            for _ in range(num_examples):
                ex_len = random.randint(1, maxlen)
                ex_str = ' '.join(map(chr, data[offset:offset+ex_len]))
                print(ex_str, file=h)
                offset += ex_len

    _create_dummy_data('train.in')
    _create_dummy_data('train.out')
    _create_dummy_data('valid.in')
    _create_dummy_data('valid.out')
    _create_dummy_data('test.in')
    _create_dummy_data('test.out')


def preprocess_translation_data(data_dir, extra_flags=None):
    preprocess_parser = preprocess.get_parser()
    preprocess_args = preprocess_parser.parse_args(
        [
            '--source-lang', 'in',
            '--target-lang', 'out',
            '--trainpref', os.path.join(data_dir, 'train'),
            '--validpref', os.path.join(data_dir, 'valid'),
            '--testpref', os.path.join(data_dir, 'test'),
            '--thresholdtgt', '0',
            '--thresholdsrc', '0',
            '--destdir', data_dir,
        ] + (extra_flags or []),
    )
    preprocess.main(preprocess_args)


def train_translation_model(data_dir, arch, extra_flags=None):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            '--task', 'translation',
            data_dir,
            '--save-dir', data_dir,
            '--arch', arch,
            '--optimizer', 'nag',
            '--lr', '0.05',
            '--max-tokens', '500',
            '--max-epoch', '1',
            '--no-progress-bar',
            '--distributed-world-size', '1',
            '--source-lang', 'in',
            '--target-lang', 'out',
        ] + (extra_flags or []),
    )
    train.main(train_args)


def generate_main(data_dir, extra_flags=None):
    generate_parser = options.get_generation_parser()
    generate_args = options.parse_args_and_arch(
        generate_parser,
        [
            data_dir,
            '--path', os.path.join(data_dir, 'checkpoint_last.pt'),
            '--beam', '3',
            '--batch-size', '64',
            '--max-len-b', '5',
            '--gen-subset', 'valid',
            '--no-progress-bar',
            '--print-alignment',
        ] + (extra_flags or []),
    )

    # evaluate model in batch mode
    generate.main(generate_args)

    # evaluate model interactively
    generate_args.buffer_size = 0
    generate_args.input = '-'
    generate_args.max_sentences = None
    orig_stdin = sys.stdin
    sys.stdin = StringIO('h e l l o\n')
    interactive.main(generate_args)
    sys.stdin = orig_stdin


def preprocess_lm_data(data_dir):
    preprocess_parser = preprocess.get_parser()
    preprocess_args = preprocess_parser.parse_args([
        '--only-source',
        '--trainpref', os.path.join(data_dir, 'train.out'),
        '--validpref', os.path.join(data_dir, 'valid.out'),
        '--testpref', os.path.join(data_dir, 'test.out'),
        '--destdir', data_dir,
    ])
    preprocess.main(preprocess_args)


def train_language_model(data_dir, arch):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            '--task', 'language_modeling',
            data_dir,
            '--arch', arch,
            '--optimizer', 'nag',
            '--lr', '0.1',
            '--criterion', 'adaptive_loss',
            '--adaptive-softmax-cutoff', '5,10,15',
            '--decoder-layers', '[(850, 3)] * 2 + [(1024,4)]',
            '--decoder-embed-dim', '280',
            '--max-tokens', '500',
            '--tokens-per-sample', '500',
            '--save-dir', data_dir,
            '--max-epoch', '1',
            '--no-progress-bar',
            '--distributed-world-size', '1',
            '--ddp-backend', 'no_c10d',
        ],
    )
    train.main(train_args)


def eval_lm_main(data_dir):
    eval_lm_parser = options.get_eval_lm_parser()
    eval_lm_args = options.parse_args_and_arch(
        eval_lm_parser,
        [
            data_dir,
            '--path', os.path.join(data_dir, 'checkpoint_last.pt'),
            '--no-progress-bar',
        ],
    )
    eval_lm.main(eval_lm_args)


if __name__ == '__main__':
    unittest.main()
