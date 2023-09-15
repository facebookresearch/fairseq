#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import contextlib
import logging
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import torch
from fairseq import checkpoint_utils, data
from omegaconf import OmegaConf


def mock_trainer(epoch, num_updates, iterations_in_epoch):
    trainer = MagicMock()
    trainer.load_checkpoint.return_value = {
        "train_iterator": {
            "epoch": epoch,
            "iterations_in_epoch": iterations_in_epoch,
            "shuffle": False,
        },
        "FakeTask": checkpoint_dict()["FakeTask"],
    }
    trainer.get_num_updates.return_value = num_updates
    trainer.task.__class__.__name__ = "FakeTask"
    trainer.task.get_checkpoint_dict.return_value = checkpoint_dict()
    trainer.task.set_checkpoint_dict = MagicMock()

    return trainer


def checkpoint_dict():
    return {
        "FakeTask": {
            "observer_stats": {
                (
                    4,
                    16,
                    "MovingAveragePerChannelMinMax",
                    "MovingAveragePerChannelMinMax",
                ): {"mod1": 1, "mod2": 2, "mod3": 3}
            }
        }
    }


def mock_dict():
    d = MagicMock()
    d.pad.return_value = 1
    d.eos.return_value = 2
    d.unk.return_value = 3
    return d


def get_trainer_and_epoch_itr(epoch, epoch_size, num_updates, iterations_in_epoch):
    tokens = torch.LongTensor(list(range(epoch_size))).view(1, -1)
    tokens_ds = data.TokenBlockDataset(
        tokens,
        sizes=[tokens.size(-1)],
        block_size=1,
        pad=0,
        eos=1,
        include_targets=False,
    )
    trainer = mock_trainer(epoch, num_updates, iterations_in_epoch)
    dataset = data.LanguagePairDataset(
        tokens_ds, tokens_ds.sizes, mock_dict(), shuffle=False
    )
    epoch_itr = data.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=[[i] for i in range(epoch_size)],
    )
    return trainer, epoch_itr


def get_mock_cfg(finetune_from_model):
    cfg_mock = OmegaConf.create(
        {
            "checkpoint": {
                "save_dir": None,
                "optimizer_overrides": "{}",
                "reset_dataloader": False,
                "reset_meters": False,
                "reset_optimizer": False,
                "reset_lr_scheduler": False,
                "finetune_from_model": finetune_from_model,
                "model_parallel_size": 1,
                "restore_file": "checkpoint_last.pt",
                "no_save": False,
                "save_interval_updates": 0,
                "no_last_checkpoints": False,
                "keep_interval_updates": 0,
                "keep_last_epochs": 0,
                "keep_best_checkpoints": 0,
            },
            "common": {
                "model_parallel_size": 1,
            },
        }
    )
    return cfg_mock


class TestCheckpointsForTaskLevelAttributes(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg_mock = get_mock_cfg(None)
        self.patches = {
            "os.makedirs": MagicMock(),
            "os.path.join": MagicMock(),
            "os.path.isfile": MagicMock(return_value=True),
            "os.path.isabs": MagicMock(return_value=False),
            "fairseq.file_io.PathManager.exists": MagicMock(return_value=False),
        }
        self.applied_patches = [patch(p, d) for p, d in self.patches.items()]
        [p.start() for p in self.applied_patches]
        logging.disable(logging.CRITICAL)

        self.trainer, self.epoch_itr = get_trainer_and_epoch_itr(2, 150, 200, 50)
        self.trainer.get_train_iterator = MagicMock(return_value=self.epoch_itr)
        self.epoch_itr.next_epoch_itr(shuffle=False)

        checkpoint_utils.save_checkpoint(
            self.cfg_mock.checkpoint, self.trainer, self.epoch_itr, None
        )

    def tearDown(self):
        patch.stopall()
        logging.disable(logging.NOTSET)

    def test_verify_checkpoint(self) -> None:
        cp_dict = self.trainer.task.get_checkpoint_dict()
        self.assertTrue(len(cp_dict) == 1)
        self.assertTrue("FakeTask" in cp_dict)
        self.assertTrue("observer_stats" in cp_dict["FakeTask"])
        self.assertTrue(len(cp_dict["FakeTask"]["observer_stats"]) == 1)
        self.assertTrue(
            (
                4,
                16,
                "MovingAveragePerChannelMinMax",
                "MovingAveragePerChannelMinMax",
            )
            in cp_dict["FakeTask"]["observer_stats"]
        )
        self.assertTrue(
            cp_dict["FakeTask"]["observer_stats"][
                (
                    4,
                    16,
                    "MovingAveragePerChannelMinMax",
                    "MovingAveragePerChannelMinMax",
                )
            ]
            == {"mod1": 1, "mod2": 2, "mod3": 3}
        )

    def test_load_checkpoint(self) -> None:
        with contextlib.redirect_stdout(StringIO()):
            # Now, load checkpoint to ensure the respective logic works as expected
            _, epoch_itr = checkpoint_utils.load_checkpoint(
                self.cfg_mock.checkpoint, self.trainer
            )

            self.trainer.task.set_checkpoint_dict.assert_called_once_with(
                checkpoint_dict()["FakeTask"]
            )


if __name__ == "__main__":
    unittest.main()
