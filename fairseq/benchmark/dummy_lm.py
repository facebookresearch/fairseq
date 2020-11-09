# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq.data import Dictionary, FairseqDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II


logger = logging.getLogger(__name__)


@dataclass
class DummyLMConfig(FairseqDataclass):
    dict_size: int = 49996
    dataset_size: int = 100000
    tokens_per_sample: int = field(
        default=512, metadata={"help": "max sequence length"}
    )
    add_bos_token: bool = False
    batch_size: Optional[int] = II("dataset.batch_size")
    max_tokens: Optional[int] = II("dataset.max_tokens")
    max_target_positions: int = II("task.tokens_per_sample")


@register_task("dummy_lm", dataclass=DummyLMConfig)
class DummyLMTask(FairseqTask):

    def __init__(self, cfg: DummyLMConfig):
        super().__init__(cfg)

        # load dictionary
        self.dictionary = Dictionary()
        for i in range(cfg.dict_size):
            self.dictionary.add_symbol("word{}".format(i))
        self.dictionary.pad_to_multiple_(8)  # often faster if divisible by 8
        logger.info("dictionary: {} types".format(len(self.dictionary)))

        seq = torch.arange(cfg.tokens_per_sample + 1) + self.dictionary.pad() + 1

        self.dummy_src = seq[:-1]
        self.dummy_tgt = seq[1:]

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.cfg.batch_size is not None:
            bsz = self.cfg.batch_size
        else:
            bsz = max(1, self.cfg.max_tokens // self.cfg.tokens_per_sample)
        self.datasets[split] = DummyDataset(
            {
                "id": 1,
                "net_input": {
                    "src_tokens": torch.stack([self.dummy_src for _ in range(bsz)]),
                    "src_lengths": torch.full(
                        (bsz,), self.cfg.tokens_per_sample, dtype=torch.long
                    ),
                },
                "target": torch.stack([self.dummy_tgt for _ in range(bsz)]),
                "nsentences": bsz,
                "ntokens": bsz * self.cfg.tokens_per_sample,
            },
            num_items=self.cfg.dataset_size,
            item_size=self.cfg.tokens_per_sample,
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class DummyDataset(FairseqDataset):
    def __init__(self, batch, num_items, item_size):
        super().__init__()
        self.batch = batch
        self.num_items = num_items
        self.item_size = item_size

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.num_items

    def collater(self, samples):
        return self.batch

    @property
    def sizes(self):
        return np.array([self.item_size] * self.num_items)

    def num_tokens(self, index):
        return self.item_size

    def size(self, index):
        return self.item_size

    def ordered_indices(self):
        return np.arange(self.num_items)

    @property
    def supports_prefetch(self):
        return False
