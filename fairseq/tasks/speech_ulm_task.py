# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from fairseq.data import Dictionary
from fairseq.data.codedataset import ExpressiveCodeDataConfig, CodeDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING, DictConfig


logger = logging.getLogger(__name__)


class UnitDictionary(Dictionary):
    """
    A fixed-sized Dictionary that operates on integer-valued tokens
    wth a trivial (identity) token <-> id mapping.
    Special symbols (bos, eos, ...) have ids above n_units.
    """

    def __init__(
        self,
        *,  # begin keyword-only arguments
        n_units,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
        clip=False,
    ):
        self.n_units = n_units
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.clip = clip

        self.symbols = []
        self.count = []
        self.indices = {}
        for i in range(n_units):
            self.add_symbol(str(i))

        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)

        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def encode_line(self, line, append_eos=True, prepend_bos=False) -> torch.IntTensor:
        words = [int(x) for x in line.split()]
        if self.clip:
            words = [min(self.n_units - 1, word) for word in words]
        if prepend_bos:
            words = [self.bos_index] + words
        if append_eos:
            words.append(self.eos_index)
        ids = torch.IntTensor(words)
        return ids


@dataclass
class SpeechUnitModelingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "Path to data config.json"})
    max_token_duration: int = field(
        default=20, metadata={"help": "all token durations are capped to this value"}
    )
    tokens_per_sample: int = field(
        default=1024, metadata={"help": "tokens in a sample"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max target positions"}
    )

    # duration modeling
    ignore_duration_input: bool = field(
        default=False, metadata={"help": "whether token durations should be zeroed out"}
    )
    discrete_duration: bool = field(
        default=False, metadata={"help": "treat duration as discrete variable"}
    )
    # F0 modeling
    ignore_f0_input: bool = field(
        default=False, metadata={"help": "whether F0 should be zeroed out"}
    )
    discrete_f0: bool = field(
        default=False, metadata={"help": "load quantized f0. get bin from config"}
    )
    log_f0: bool = field(
        default=False, metadata={"help": "whether f0 should be modeled in log space"}
    )
    normalize_f0_mean: bool = field(
        default=False, metadata={"help": "whether normalize f0 by speaker mean"}
    )
    normalize_f0_std: bool = field(
        default=False, metadata={"help": "whether normalize f0 by speaker stddev"}
    )
    interpolate_f0: bool = field(
        default=False,
        metadata={"help": "whether interpolate f0 for non-voiced segments"},
    )

    # input/output streams
    stream_shifts: str = field(
        default="0,0",
        metadata={
            "help": (
                "comma-separated integer list denoting right-shift for "
                "duration and pitch streams"
            )
        },
    )


@register_task("speech_unit_modeling", dataclass=SpeechUnitModelingConfig)
class SpeechUnitLanguageModelingTask(FairseqTask):
    def __init__(self, cfg: SpeechUnitModelingConfig) -> None:
        super().__init__(cfg)
        assert not self.cfg.normalize_f0_std or self.cfg.normalize_f0_mean

        self.data_config = ExpressiveCodeDataConfig(cfg.data)
        self._source_dictionary = self._target_dictionary = UnitDictionary(
            n_units=self.data_config.n_units
        )
        self._source_duration_dictionary = self._target_duration_dictionary = (
            UnitDictionary(n_units=self.cfg.max_token_duration + 1, clip=True)
            if self.cfg.discrete_duration
            else None
        )
        self._source_f0_dictionary = self._target_f0_dictionary = (
            UnitDictionary(n_units=self.data_config.f0_vq_n_units)
            if self.cfg.discrete_f0
            else None
        )

        self._channel_names = ["token", "duration", "f0"]
        self._channel_sizes = [
            len(self.target_dictionary),
            len(self.target_duration_dictionary) if self.cfg.discrete_duration else 1,
            len(self.target_f0_dictionary) if self.cfg.discrete_f0 else 1,
        ]

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return self._source_dictionary

    @property
    def source_duration_dictionary(self) -> Optional[Dictionary]:
        return self._source_duration_dictionary

    @property
    def source_f0_dictionary(self) -> Optional[Dictionary]:
        return self._source_f0_dictionary

    @property
    def channel_names(self) -> List[str]:
        return self._channel_names

    @property
    def channel_sizes(self) -> List[int]:
        return self._channel_sizes

    @property
    def dictionary(self) -> Optional[Dictionary]:
        return self._source_dictionary

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self._target_dictionary

    @property
    def target_duration_dictionary(self) -> Optional[Dictionary]:
        return self._target_duration_dictionary

    @property
    def target_f0_dictionary(self) -> Optional[Dictionary]:
        return self._target_f0_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return [self._dictionaries[l] for l in self.cfg.labels]

    @classmethod
    def setup_task(
        cls, cfg: SpeechUnitModelingConfig, **kwargs
    ) -> "SpeechUnitLanguageModelingTask":
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        self.datasets[split] = CodeDataset(
            manifest=self.data_config.manifests[split],
            dictionary=self.source_dictionary,
            dur_dictionary=self.source_duration_dictionary,
            f0_dictionary=self.source_f0_dictionary,
            config=self.data_config,
            discrete_dur=self.cfg.discrete_duration,
            discrete_f0=self.cfg.discrete_f0,
            log_f0=self.cfg.log_f0,
            normalize_f0_mean=self.cfg.normalize_f0_mean,
            normalize_f0_std=self.cfg.normalize_f0_std,
            interpolate_f0=self.cfg.interpolate_f0,
            shifts=self.cfg.stream_shifts,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def build_criterion(self, cfg: DictConfig):
        import fairseq.criterions

        return fairseq.criterions.build_criterion(cfg, self)
