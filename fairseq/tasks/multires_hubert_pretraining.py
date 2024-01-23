# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )


@dataclass
class MultiresHubertPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr50", "ltr25"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    #     label_rate: 1,2,2,5
    #                 (imply (1,2), (2,5))
    #     if base label_rate = 50
    #     (1,2), (2,5) --> label rates 50, 25, 10
    label_rate_ratios: List[int] = field(default=MISSING, metadata={"help": "tuple for label rates e.g., [(1,2), (2,5)]"})
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )


@register_task("multires_hubert_pretraining", dataclass=MultiresHubertPretrainingConfig)
class MultiresHubertPretrainingTask(FairseqTask):
    """
    Multiresolution HuBERT Pretraining Task.
    The task is based on `HubertPretrainingTask` but extended to multiresolution.
    """

    cfg: MultiresHubertPretrainingConfig

    def __init__(
        self,
        cfg: MultiresHubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"MultiresHubertPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
            self.res_number = 1
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: MultiresHubertPretrainingConfig, **kwargs
    ) -> "MultiresHubertPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        self.res_number = len(label_dir)
        dictionaries = [ (Dictionary.load(f"{label_dir}/dict.{label}.txt") if label is not "" else None ) for label in self.cfg.labels]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [(dict.pad() if dict is not None else None) for dict in dicts]
        eos_list = [(dict.eos() if dict is not None else None) for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [(f"{self.get_label_dir()}/{split}.{l}" if l != "" else None) for l in self.cfg.labels]

        base_rate = self.cfg.label_rate
        self.label_rates = [base_rate]
        label_rate_ratios = self.cfg.label_rate_ratios
        self.label_rate_ratios = []
        for i in range(len(label_rate_ratios) // 2):

            upsample_rate, downsample_rate = label_rate_ratios[i * 2], label_rate_ratios[i * 2 + 1]
            # parse label rate ratios
            self.label_rate_ratios.append((upsample_rate, downsample_rate))
            base_rate = base_rate * upsample_rate // downsample_rate
            self.label_rates.append(base_rate)

        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = HubertDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.label_rates,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
