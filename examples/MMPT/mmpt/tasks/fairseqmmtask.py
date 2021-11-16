# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
make a general fairseq task for MM pretraining.
"""

import random

from fairseq.tasks import LegacyFairseqTask, register_task

from .task import Task
from .retritask import RetriTask
from ..datasets import FairseqMMDataset
from .. import utils


@register_task("mmtask")
class FairseqMMTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument(
            "taskconfig",
            metavar="FILE",
            help=(
                "taskconfig to load all configurations"
                "outside fairseq parser."),
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        return FairseqMMTask(args)

    def __init__(self, args):
        super().__init__(args)
        config = utils.load_config(args)
        self.mmtask = Task.config_task(config)
        self.mmtask.build_dataset()
        self.mmtask.build_model()
        self.mmtask.build_loss()

    def load_dataset(self, split, **kwargs):
        split_map = {
            "train": self.mmtask.train_data,
            "valid": self.mmtask.val_data,
            "test": self.mmtask.test_data,
        }
        if split not in split_map:
            raise ValueError("unknown split type.")
        if split_map[split] is not None:
            self.datasets[split] = FairseqMMDataset(split_map[split])

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        random.seed(epoch)
        if dataset.mmdataset.split == "train" \
                and isinstance(self.mmtask, RetriTask):
            if epoch >= self.mmtask.config.retri_epoch:
                if not hasattr(self.mmtask, "retri_dataloader"):
                    self.mmtask.build_dataloader()
                self.mmtask.retrive_candidates(epoch)

        return super().get_batch_iterator(
            dataset, max_tokens, max_sentences, max_positions,
            ignore_invalid_inputs, required_batch_size_multiple,
            seed, num_shards, shard_id, num_workers, epoch,
            data_buffer_size, disable_iterator_cache,
            grouped_shuffling, update_epoch_batch_itr)

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None
