# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from io import StringIO
import os
import random
import sys
import tempfile
import unittest

import torch

from fairseq import options

import preprocess, train, generate, interactive


class TestBinaries(unittest.TestCase):

    def test_binaries(self):
        # comment this out to debug the unittest if it's failing
        self.mock_stdout()

        with tempfile.TemporaryDirectory() as data_dir:
            self.create_dummy_data(data_dir)
            self.preprocess_data(data_dir)
            self.train_model(data_dir)
            self.generate(data_dir)

        self.unmock_stdout()

    def create_dummy_data(self, data_dir, num_examples=1000, maxlen=20):

        def _create_dummy_data(filename):
            data = torch.rand(num_examples * maxlen)
            data = 97 + torch.floor(26 * data).int()
            with open(os.path.join(data_dir, filename), 'w') as h:
                offset = 0
                for i in range(num_examples):
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

    def preprocess_data(self, data_dir):
        preprocess_parser = preprocess.get_parser()
        preprocess_args = preprocess_parser.parse_args([
            '--source-lang', 'in',
            '--target-lang', 'out',
            '--trainpref', os.path.join(data_dir, 'train'),
            '--validpref', os.path.join(data_dir, 'valid'),
            '--testpref', os.path.join(data_dir, 'test'),
            '--thresholdtgt', '0',
            '--thresholdsrc', '0',
            '--destdir', data_dir,
        ])
        preprocess.main(preprocess_args)

    def train_model(self, data_dir):
        train_parser = options.get_training_parser()
        train_args = options.parse_args_and_arch(
            train_parser,
            [
                data_dir,
                '--arch', 'fconv_iwslt_de_en',
                '--optimizer', 'nag',
                '--lr', '0.05',
                '--max-tokens', '500',
                '--save-dir', data_dir,
                '--max-epoch', '1',
                '--no-progress-bar',
            ],
        )
        train.main(train_args)

    def generate(self, data_dir):
        generate_parser = options.get_generation_parser()
        generate_args = generate_parser.parse_args([
            data_dir,
            '--path', os.path.join(data_dir, 'checkpoint_best.pt'),
            '--beam', '5',
            '--batch-size', '32',
            '--gen-subset', 'valid',
            '--no-progress-bar',
        ])

        # evaluate model in batch mode
        generate.main(generate_args)

        # evaluate model interactively
        orig_stdin = sys.stdin
        sys.stdin = StringIO('h e l l o\n')
        interactive.main(generate_args)
        sys.stdin = orig_stdin

    def mock_stdout(self):
        self._orig_stdout = sys.stdout
        sys.stdout = StringIO()

    def unmock_stdout(self):
        if hasattr(self, '_orig_stdout'):
            sys.stdout = self._orig_stdout


if __name__ == '__main__':
    unittest.main()
