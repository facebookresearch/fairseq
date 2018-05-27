# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import unittest

import itertools
from unittest.mock import MagicMock, patch

import train


def mock_trainer(epoch, num_updates):
    trainer = MagicMock()
    trainer.load_checkpoint.return_value = {'epoch': epoch}
    trainer.get_num_updates.return_value = num_updates
    return trainer


def mock_loader(length):
    loader = MagicMock()
    loader.__next__.return_value = list(range(length))
    return loader


class TestLoadCheckpoint(unittest.TestCase):

    def setUp(self):
        self.patches = {
            'os.makedirs': MagicMock(),
            'os.path.join': MagicMock(),
            'os.path.isfile': MagicMock(return_value=True),
        }
        self.applied_patches = [patch(p, d) for p, d in self.patches.items()]
        [p.start() for p in self.applied_patches]

    def test_load_partial_checkpoint(self):
        trainer = mock_trainer(2, 200)
        loader = mock_loader(150)
        epoch, ds = train.load_checkpoint(MagicMock(), trainer, loader)
        self.assertEqual(epoch, 2)
        self.assertEqual(next(ds), 50)

    def test_load_full_checkpoint(self):
        trainer = mock_trainer(2, 150)
        loader = mock_loader(150)
        epoch, ds = train.load_checkpoint(MagicMock(), trainer, loader)
        self.assertEqual(epoch, 2)
        self.assertEqual(next(iter(ds)), 0)

    def test_load_no_checkpoint(self):
        trainer = mock_trainer(0, 0)
        loader = mock_loader(150)
        self.patches['os.path.isfile'].return_value = False

        epoch, ds = train.load_checkpoint(MagicMock(), trainer, loader)
        self.assertEqual(epoch, 1)
        self.assertEqual(next(iter(ds)), 0)

    def tearDown(self):
        patch.stopall()


if __name__ == '__main__':
    unittest.main()
