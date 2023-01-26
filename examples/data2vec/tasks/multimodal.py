# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import sys

from dataclasses import dataclass
from typing import Optional, List
from omegaconf import II

from fairseq.data.iterators import GroupedEpochBatchIterator

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask
from fairseq.tasks.masked_lm import MaskedLMConfig, MaskedLMTask
from .mae_image_pretraining import MaeImagePretrainingConfig, MaeImagePretrainingTask
from examples.data2vec.data.modality import Modality

from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    ModalityDatasetItem,
)


@dataclass
class MultimodalPretrainingConfig(FairseqDataclass):
    audio: Optional[AudioPretrainingConfig] = None
    image: Optional[MaeImagePretrainingConfig] = None
    text: Optional[MaskedLMConfig] = None

    audio_ratio: float = 1
    image_ratio: float = 1
    text_ratio: float = 1

    max_tokens: Optional[int] = II("dataset.max_tokens")
    batch_size: Optional[int] = II("dataset.batch_size")
    update_freq: List[int] = II("optimization.update_freq")

    rebuild_batches: bool = True


@register_task("multimodal_pretraining", dataclass=MultimodalPretrainingConfig)
class MultimodalPretrainingTask(FairseqTask):
    """ """

    cfg: MultimodalPretrainingConfig

    def __init__(self, cfg: MultimodalPretrainingConfig):
        super().__init__(cfg)
        self.audio_task = (
            AudioPretrainingTask(cfg.audio) if cfg.audio is not None else None
        )
        self.image_task = (
            MaeImagePretrainingTask(cfg.image) if cfg.image is not None else None
        )
        self.text_task = MaskedLMTask(cfg.text) if cfg.text is not None else None

        self.mult_ratios = []

    @classmethod
    def setup_task(cls, cfg: MultimodalPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        datasets = []
        self.mult_ratios = []

        def load_ds(task, name, ratio):
            if task is not None:
                task.load_dataset(split)
                ds = ModalityDatasetItem(
                    datasetname=name,
                    dataset=task.dataset(split),
                    max_positions=task.max_positions(),
                    max_tokens=self.cfg.max_tokens,
                    max_sentences=self.cfg.batch_size,
                )
                datasets.append(ds)
                self.mult_ratios.append(ratio)

        load_ds(self.audio_task, Modality.AUDIO, self.cfg.audio_ratio)
        load_ds(self.image_task, Modality.IMAGE, self.cfg.image_ratio)
        load_ds(self.text_task, Modality.TEXT, self.cfg.text_ratio)

        assert len(datasets) > 0

        self.datasets[split] = MultiModalityDataset(datasets)

    @property
    def supported_modalities(self):
        modalities = []
        if self.cfg.text is not None:
            modalities.append(Modality.TEXT)
        if self.cfg.audio is not None:
            modalities.append(Modality.AUDIO)
        if self.cfg.image is not None:
            modalities.append(Modality.IMAGE)

        return modalities

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
        epoch=0,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            self.mult_ratios, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=max(self.cfg.update_freq),
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize
