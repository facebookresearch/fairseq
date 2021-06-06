# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import II

from .dummy_dataset import DummyDataset
from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)


@dataclass
class DummyMaskedLMConfig(FairseqDataclass):
    dict_size: int = 49996
    dataset_size: int = 100000
    tokens_per_sample: int = field(
        default=512,
        metadata={
            "help": "max number of total tokens over all"
            " segments per sample for BERT dataset"
        },
    )
    batch_size: Optional[int] = II("dataset.batch_size")
    max_tokens: Optional[int] = II("dataset.max_tokens")
    max_target_positions: int = II("task.tokens_per_sample")


@register_task("dummy_masked_lm", dataclass=DummyMaskedLMConfig)
class DummyMaskedLMTask(FairseqTask):
    def __init__(self, cfg: DummyMaskedLMConfig):
        super().__init__(cfg)

        self.dictionary = Dictionary()
        for i in range(cfg.dict_size):
            self.dictionary.add_symbol("word{}".format(i))
        logger.info("dictionary: {} types".format(len(self.dictionary)))
        # add mask token
        self.mask_idx = self.dictionary.add_symbol("<mask>")
        self.dictionary.pad_to_multiple_(8)  # often faster if divisible by 8

        mask_idx = 0
        pad_idx = 1
        seq = torch.arange(cfg.tokens_per_sample) + pad_idx + 1
        mask = torch.arange(2, cfg.tokens_per_sample, 7)  # ~15%
        src = seq.clone()
        src[mask] = mask_idx
        tgt = torch.full_like(seq, pad_idx)
        tgt[mask] = seq[mask]

        self.dummy_src = src
        self.dummy_tgt = tgt

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
