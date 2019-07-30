# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
                preprocess_translation_data(data_dir, ['--dataset-impl', 'raw'])
                train_translation_model(data_dir, 'fconv_iwslt_de_en', ['--dataset-impl', 'raw'])
                generate_main(data_dir, ['--dataset-impl', 'raw'])

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
                    '--temperature', '2',
                    '--beam', '2',
                    '--nbest', '2',
                ])
                generate_main(data_dir, [
                    '--sampling',
                    '--sampling-topk', '3',
                    '--beam', '2',
                    '--nbest', '2',
                ])
                generate_main(data_dir, [
                    '--sampling',
                    '--sampling-topp', '0.2',
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
                    '--encoder-embed-dim', '8',
                    '--decoder-embed-dim', '8',
                    '--decoder-out-embed-dim', '8',
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
                    '--encoder-hidden-size', '16',
                    '--encoder-embed-dim', '8',
                    '--decoder-embed-dim', '8',
                    '--decoder-out-embed-dim', '8',
                    '--decoder-layers', '2',
                ])
                generate_main(data_dir)

    def test_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_transformer') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'transformer_iwslt_de_en', [
                    '--encoder-layers', '2',
                    '--decoder-layers', '2',
                    '--encoder-embed-dim', '8',
                    '--decoder-embed-dim', '8',
                ])
                generate_main(data_dir)

    def test_lightconv(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_lightconv') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'lightconv_iwslt_de_en', [
                    '--encoder-conv-type', 'lightweight',
                    '--decoder-conv-type', 'lightweight',
                    '--encoder-embed-dim', '8',
                    '--decoder-embed-dim', '8',
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
                    '--encoder-embed-dim', '8',
                    '--decoder-embed-dim', '8',
                ])
                generate_main(data_dir)

    def test_mixture_of_experts(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_moe') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, 'transformer_iwslt_de_en', [
                    '--task', 'translation_moe',
                    '--method', 'hMoElp',
                    '--mean-pool-gating-network',
                    '--num-experts', '3',
                    '--encoder-layers', '2',
                    '--decoder-layers', '2',
                    '--encoder-embed-dim', '8',
                    '--decoder-embed-dim', '8',
                ])
                generate_main(data_dir, [
                    '--task', 'translation_moe',
                    '--method', 'hMoElp',
                    '--mean-pool-gating-network',
                    '--num-experts', '3',
                    '--gen-expert', '0'
                ])


class TestStories(unittest.TestCase):

    def test_fconv_self_att_wp(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_fconv_self_att_wp') as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                config = [
                    '--encoder-layers', '[(128, 3)] * 2',
                    '--decoder-layers', '[(128, 3)] * 2',
                    '--decoder-attention', 'True',
                    '--encoder-attention', 'False',
                    '--gated-attention', 'True',
                    '--self-attention', 'True',
                    '--project-input', 'True',
                    '--encoder-embed-dim', '8',
                    '--decoder-embed-dim', '8',
                    '--decoder-out-embed-dim', '8',
                    '--multihead-self-attention-nheads', '2'
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
                train_language_model(data_dir, 'fconv_lm', [
                    '--decoder-layers', '[(850, 3)] * 2 + [(1024,4)]',
                    '--decoder-embed-dim', '280',
                    '--optimizer', 'nag',
                    '--lr', '0.1',
                ])
                eval_lm_main(data_dir)

    def test_transformer_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_transformer_lm') as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_language_model(data_dir, 'transformer_lm', ['--add-bos-token'])
                eval_lm_main(data_dir)


class TestMaskedLanguageModel(unittest.TestCase):

    def test_legacy_masked_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_legacy_mlm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_legacy_masked_language_model(data_dir, "masked_lm")

    def _test_pretrained_masked_lm_for_translation(self, learned_pos_emb, encoder_only):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_mlm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_legacy_masked_language_model(
                    data_dir,
                    arch="masked_lm",
                    extra_args=('--encoder-learned-pos',) if learned_pos_emb else ()
                )
                with tempfile.TemporaryDirectory(
                    "test_mlm_translation"
                ) as translation_dir:
                    create_dummy_data(translation_dir)
                    preprocess_translation_data(
                        translation_dir, extra_flags=["--joined-dictionary"]
                    )
                    # Train transformer with data_dir/checkpoint_last.pt
                    train_translation_model(
                        translation_dir,
                        arch="transformer_from_pretrained_xlm",
                        extra_flags=[
                            "--decoder-layers",
                            "1",
                            "--decoder-embed-dim",
                            "32",
                            "--decoder-attention-heads",
                            "1",
                            "--decoder-ffn-embed-dim",
                            "32",
                            "--encoder-layers",
                            "1",
                            "--encoder-embed-dim",
                            "32",
                            "--encoder-attention-heads",
                            "1",
                            "--encoder-ffn-embed-dim",
                            "32",
                            "--pretrained-xlm-checkpoint",
                            "{}/checkpoint_last.pt".format(data_dir),
                            "--activation-fn",
                            "gelu",
                            "--max-source-positions",
                            "500",
                            "--max-target-positions",
                            "500",
                        ] + (
                            ["--encoder-learned-pos", "--decoder-learned-pos"]
                            if learned_pos_emb else []
                        ) + (['--init-encoder-only'] if encoder_only else []),
                        task="translation_from_pretrained_xlm",
                    )

    def test_pretrained_masked_lm_for_translation_learned_pos_emb(self):
        self._test_pretrained_masked_lm_for_translation(True, False)

    def test_pretrained_masked_lm_for_translation_sinusoidal_pos_emb(self):
        self._test_pretrained_masked_lm_for_translation(False, False)

    def test_pretrained_masked_lm_for_translation_encoder_only(self):
        self._test_pretrained_masked_lm_for_translation(True, True)


def train_legacy_masked_language_model(data_dir, arch, extra_args=()):
    train_parser = options.get_training_parser()
    # TODO: langs should be in and out right?
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            "cross_lingual_lm",
            data_dir,
            "--arch",
            arch,
            # Optimizer args
            "--optimizer",
            "adam",
            "--lr-scheduler",
            "reduce_lr_on_plateau",
            "--lr-shrink",
            "0.5",
            "--lr",
            "0.0001",
            "--min-lr",
            "1e-09",
            # dropout, attention args
            "--dropout",
            "0.1",
            "--attention-dropout",
            "0.1",
            # MLM args
            "--criterion",
            "legacy_masked_lm_loss",
            "--masked-lm-only",
            "--monolingual-langs",
            "in,out",
            "--num-segment",
            "5",
            # Transformer args: use a small transformer model for fast training
            "--encoder-layers",
            "1",
            "--encoder-embed-dim",
            "32",
            "--encoder-attention-heads",
            "1",
            "--encoder-ffn-embed-dim",
            "32",
            # Other training args
            "--max-tokens",
            "500",
            "--tokens-per-sample",
            "500",
            "--save-dir",
            data_dir,
            "--max-epoch",
            "1",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--dataset-impl",
            "raw",
        ] + list(extra_args),
    )
    train.main(train_args)


class TestCommonOptions(unittest.TestCase):

    def test_optimizers(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_optimizers') as data_dir:
                # Use just a bit of data and tiny model to keep this test runtime reasonable
                create_dummy_data(data_dir, num_examples=10, maxlen=5)
                preprocess_translation_data(data_dir)
                optimizers = ['adafactor', 'adam', 'nag', 'adagrad', 'sgd', 'adadelta']
                last_checkpoint = os.path.join(data_dir, 'checkpoint_last.pt')
                for optimizer in optimizers:
                    if os.path.exists(last_checkpoint):
                        os.remove(last_checkpoint)
                    train_translation_model(data_dir, 'lstm', [
                        '--required-batch-size-multiple', '1',
                        '--encoder-layers', '1',
                        '--encoder-hidden-size', '32',
                        '--decoder-layers', '1',
                        '--optimizer', optimizer,
                    ])
                    generate_main(data_dir)


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
    preprocess_parser = options.get_preprocessing_parser()
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


def train_translation_model(data_dir, arch, extra_flags=None, task='translation'):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            '--task', task,
            data_dir,
            '--save-dir', data_dir,
            '--arch', arch,
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
    preprocess_parser = options.get_preprocessing_parser()
    preprocess_args = preprocess_parser.parse_args([
        '--only-source',
        '--trainpref', os.path.join(data_dir, 'train.out'),
        '--validpref', os.path.join(data_dir, 'valid.out'),
        '--testpref', os.path.join(data_dir, 'test.out'),
        '--destdir', data_dir,
    ])
    preprocess.main(preprocess_args)


def train_language_model(data_dir, arch, extra_flags=None):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            '--task', 'language_modeling',
            data_dir,
            '--arch', arch,
            '--optimizer', 'adam',
            '--lr', '0.0001',
            '--criterion', 'adaptive_loss',
            '--adaptive-softmax-cutoff', '5,10,15',
            '--max-tokens', '500',
            '--tokens-per-sample', '500',
            '--save-dir', data_dir,
            '--max-epoch', '1',
            '--no-progress-bar',
            '--distributed-world-size', '1',
            '--ddp-backend', 'no_c10d',
        ] + (extra_flags or []),
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
