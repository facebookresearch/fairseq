# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import torch

from fairseq import data, checkpoint_utils


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
    tokens = torch.LongTensor(list(range(epoch_size))).view(1, -1)
    tokens_ds = data.TokenBlockDataset(
        tokens, sizes=[tokens.size(-1)], block_size=1, pad=0, eos=1, include_targets=False,
    )
    trainer = mock_trainer(epoch, num_updates, iterations_in_epoch)
    dataset = data.LanguagePairDataset(tokens_ds, tokens_ds.sizes, mock_dict(), shuffle=False)
    epoch_itr = data.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=[[i] for i in range(epoch_size)],
    )
    return trainer, epoch_itr


def get_mock_args(finetune_from_model=None):
    args_mock = MagicMock()
    args_mock.optimizer_overrides = '{}'
    args_mock.reset_dataloader = False
    args_mock.reset_meters = False
    args_mock.reset_optimizer = False
    args_mock.reset_lr_scheduler = False
    args_mock.finetune_from_model = finetune_from_model
    args_mock.model_parallel_size = 1
    return args_mock


class TestLoadCheckpoint(unittest.TestCase):

    def setUp(self):
        self.args_mock = get_mock_args()
        self.patches = {
            'os.makedirs': MagicMock(),
            'os.path.join': MagicMock(),
            'os.path.isfile': MagicMock(return_value=True),
            'os.path.isabs': MagicMock(return_value=False),
            'fairseq.file_io.PathManager.exists': MagicMock(return_value=False),
        }
        self.applied_patches = [patch(p, d) for p, d in self.patches.items()]
        [p.start() for p in self.applied_patches]
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        patch.stopall()
        logging.disable(logging.NOTSET)

    def test_load_partial_checkpoint(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(2, 150, 200, 50)
            trainer.get_train_iterator = MagicMock(return_value=epoch_itr)

            _, epoch_itr = checkpoint_utils.load_checkpoint(self.args_mock, trainer)

            self.assertEqual(epoch_itr.epoch, 2)
            self.assertEqual(epoch_itr.iterations_in_epoch, 50)

            itr = epoch_itr.next_epoch_itr(shuffle=False)
            self.assertEqual(epoch_itr.epoch, 2)
            self.assertEqual(epoch_itr.iterations_in_epoch, 50)

            self.assertEqual(next(itr)['net_input']['src_tokens'][0].item(), 50)
            self.assertEqual(epoch_itr.iterations_in_epoch, 51)

            for _ in range(150 - 52):
                next(itr)
            self.assertEqual(epoch_itr.iterations_in_epoch, 149)
            self.assertTrue(itr.has_next())
            next(itr)
            self.assertFalse(itr.has_next())

            itr = epoch_itr.next_epoch_itr(shuffle=False)
            self.assertTrue(itr.has_next())
            self.assertEqual(epoch_itr.epoch, 3)
            self.assertEqual(epoch_itr.iterations_in_epoch, 0)

    def test_load_full_checkpoint(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(2, 150, 300, 150)
            trainer.get_train_iterator = MagicMock(return_value=epoch_itr)

            _, epoch_itr = checkpoint_utils.load_checkpoint(self.args_mock, trainer)
            itr = epoch_itr.next_epoch_itr(shuffle=False)

            self.assertEqual(epoch_itr.epoch, 3)
            self.assertEqual(epoch_itr.iterations_in_epoch, 0)
            self.assertEqual(next(itr)['net_input']['src_tokens'][0].item(), 0)

    def test_load_no_checkpoint(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(1, 150, 0, 0)
            trainer.get_train_iterator = MagicMock(return_value=epoch_itr)
            self.patches['os.path.isfile'].return_value = False

            _, epoch_itr = checkpoint_utils.load_checkpoint(self.args_mock, trainer)
            itr = epoch_itr.next_epoch_itr(shuffle=False)

            self.assertEqual(epoch_itr.epoch, 1)
            self.assertEqual(epoch_itr.iterations_in_epoch, 0)
            self.assertEqual(next(itr)['net_input']['src_tokens'][0].item(), 0)

    def test_finetune_from_model_args_conflict(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(1, 150, 0, 0)
            trainer.get_train_iterator = MagicMock(return_value=epoch_itr)

            for arg in ['reset_optimizer', 'reset_lr_scheduler', 'reset_meters', 'reset_dataloader']:
                with self.subTest(arg=arg):
                    args_mock = get_mock_args("/temp/checkpoint_pretrained.pt")
                    setattr(args_mock, arg, True)
                    with self.assertRaises(Exception) as context:
                        _, _ = checkpoint_utils.load_checkpoint(args_mock, trainer)

                    self.assertTrue(
                        "--finetune-from-model can not be set together with either --reset-optimizer"
                        " or reset_lr_scheduler or reset_meters or reset_dataloader" in str(context.exception)
                    )

    def test_finetune_from_model(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(1, 150, 0, 0)
            trainer.get_train_iterator = MagicMock(return_value=epoch_itr)
            from_model_path = "/temp/checkpoint_pretrained.pt"
            args_mock = get_mock_args(from_model_path)
            args_mock.restore_file = "checkpoint_last.pt"

            def mock_finetune_exist(path):
                if path == from_model_path:
                    return True
                else:
                    return False
            self.patches['fairseq.file_io.PathManager.exists'].side_effect = mock_finetune_exist
            _, _ = checkpoint_utils.load_checkpoint(args_mock, trainer)
            checkpoint_path, reset_optimizer, reset_lr_scheduler, \
                optimizer_overrides = trainer.load_checkpoint.call_args[0]
            reset_meters = trainer.load_checkpoint.call_args[1]['reset_meters']
            self.assertTrue(reset_optimizer)
            self.assertTrue(reset_lr_scheduler)
            self.assertTrue(reset_meters)

    def test_finetune_from_model_resume(self):
        with contextlib.redirect_stdout(StringIO()):
            trainer, epoch_itr = get_trainer_and_epoch_itr(1, 150, 0, 0)
            trainer.get_train_iterator = MagicMock(return_value=epoch_itr)
            from_model_path = "/temp/checkpoint_pretrained.pt"
            args_mock = get_mock_args(from_model_path)
            args_mock.restore_file = "checkpoint_last.pt"

            # launch second time
            # both restore_file=checkpoint_last.pt and finetune_from_model are set
            def mock_finetune_exist(path):
                if path == from_model_path or path.endsWith('checkpoint_last.pt'):
                    return True
                else:
                    return False
            self.patches['fairseq.file_io.PathManager.exists'].side_effect = mock_finetune_exist
            _, _ = checkpoint_utils.load_checkpoint(args_mock, trainer)
            checkpoint_path, reset_optimizer, reset_lr_scheduler, \
                optimizer_overrides = trainer.load_checkpoint.call_args[0]
            reset_meters = trainer.load_checkpoint.call_args[1]['reset_meters']
            self.assertFalse(reset_optimizer)
            self.assertFalse(reset_lr_scheduler)
            self.assertFalse(reset_meters)


if __name__ == '__main__':
    unittest.main()
