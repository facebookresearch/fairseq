# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
from io import StringIO
import unittest
from unittest.mock import MagicMock, patch

import torch

from fairseq import data

import train


def mock_trainer(epoch, num_updates, iterations_in_epoch):
    trainer = MagicMock()
    trainer.load_checkpoint.return_value = {
        'train_iterator': {
            'epoch': epoch,
            'iterations_in_epoch': iterations_in_epoch,
            'shuffle': False,
        },
    }
    trainer.get_num_updates.return_value = num_updates
    return trainer


def mock_dict():
    d = MagicMock()
    d.pad.return_value = 1
    d.eos.return_value = 2
    d.unk.return_value = 3
    return d


def get_trainer_and_epoch_itr(epoch, epoch_size, num_updates, iterations_in_epoch):
    tokens = torch.LongTensor(list(range(epoch_size)))
    tokens_ds = data.TokenBlockDataset(tokens, [len(tokens)], 1, include_targets=False)
    trainer = mock_trainer(epoch, num_updates, iterations_in_epoch)
    epoch_itr = data.EpochBatchIterator(
        dataset=data.LanguagePairDataset(tokens_ds, tokens_ds.sizes, mock_dict(), shuffle=False),
        batch_sampler=[[i] for i in range(epoch_size)],
    )
    return trainer, epoch_itr


class TestLoadCheckpoint(unittest.TestCase):

    def setUp(self):
        self.args_mock = MagicMock()
        self.args_mock.optimizer_overrides = '{}'
        self.patches = {
            'os.makedirs': MagicMock(),
            'os.path.join': MagicMock(),
            'os.path.isfile': MagicMock(return_value=True),
        }
        self.applied_patches = [patch(p, d) for p, d in self.patches.items()]
        [p.start() for p in self.applied_patches]


    def test_load_partial_checkpoint(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(2, 150, 200, 50)

            train.load_checkpoint(self.args_mock, trainer, epoch_itr)
            self.assertEqual(epoch_itr.epoch, 2)
            self.assertEqual(epoch_itr.iterations_in_epoch, 50)

            itr = epoch_itr.next_epoch_itr(shuffle=False)
            self.assertEqual(epoch_itr.epoch, 2)
            self.assertEqual(epoch_itr.iterations_in_epoch, 50)

            self.assertEqual(next(itr)['net_input']['src_tokens'][0].item(), 50)
            self.assertEqual(epoch_itr.iterations_in_epoch, 51)

    def test_load_full_checkpoint(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(2, 150, 300, 150)

            train.load_checkpoint(self.args_mock, trainer, epoch_itr)
            itr = epoch_itr.next_epoch_itr(shuffle=False)

            self.assertEqual(epoch_itr.epoch, 3)
            self.assertEqual(epoch_itr.iterations_in_epoch, 0)
            self.assertEqual(next(itr)['net_input']['src_tokens'][0].item(), 0)

    def test_load_no_checkpoint(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(0, 150, 0, 0)
            self.patches['os.path.isfile'].return_value = False

            train.load_checkpoint(self.args_mock, trainer, epoch_itr)
            itr = epoch_itr.next_epoch_itr(shuffle=False)

            self.assertEqual(epoch_itr.epoch, 1)
            self.assertEqual(epoch_itr.iterations_in_epoch, 0)
            self.assertEqual(next(itr)['net_input']['src_tokens'][0].item(), 0)

    def tearDown(self):
        patch.stopall()


if __name__ == '__main__':
    unittest.main()
