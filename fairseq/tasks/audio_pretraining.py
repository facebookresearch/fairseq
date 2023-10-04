# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, OrderedDict
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import BinarizedAudioDataset, FileAudioDataset, SubsampleDataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.text_compressor import TextCompressionLevel

from . import FairseqTask, register_task


logger = logging.getLogger(__name__)


@dataclass
class AudioMaskingConfig:
    feature_encoder_spec: str = II("model.modalities.audio.feature_encoder_spec")
    mask_prob: float = II("model.modalities.audio.mask_prob")
    mask_prob_adjust: float = II("model.modalities.audio.mask_prob_adjust")
    mask_length: int = II("model.modalities.audio.mask_length")
    inverse_mask: bool = II("model.modalities.audio.inverse_mask")
    mask_dropout: float = II("model.modalities.audio.mask_dropout")
    clone_batch: int = II("model.clone_batch")
    expand_adjacent: bool = False
    non_overlapping: bool = False


@dataclass
class AudioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    multi_corpus_keys: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated names for loading multi corpus datasets"})
    multi_corpus_sampling_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated string of sampling weights corresponding to the multi_corpus_keys"})
    binarized_dataset: bool = field(
        default=False,
        metadata={
            "help": "if true, loads binarized dataset (useful for very large datasets). "
            "See examples/wav2vec/scripts/binarize_manifest.sh"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    tpu: bool = II("common.tpu")
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )

    rebuild_batches: bool = True
    precompute_mask_config: Optional[AudioMaskingConfig] = None

    post_save_script: Optional[str] = None

    subsample: float = 1
    seed: int = II("common.seed")


@register_task("audio_pretraining", dataclass=AudioPretrainingConfig)
class AudioPretrainingTask(FairseqTask):
    """ """

    cfg: AudioPretrainingConfig

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        compute_mask = getattr(task_cfg, "precompute_mask_config", None) is not None
        mask_args = {}
        if compute_mask:
            mask_args = task_cfg.precompute_mask_config

        if getattr(task_cfg, "binarized_dataset", False):
            self.datasets[split] = BinarizedAudioDataset(
                data_path,
                split=split,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask=compute_mask,
                **mask_args,
            )
        else:
            if task_cfg.multi_corpus_keys is None:
                manifest_path = os.path.join(data_path, "{}.tsv".format(split))                

                self.datasets[split] = FileAudioDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    pad=task_cfg.labels is not None or task_cfg.enable_padding,
                    normalize=task_cfg.normalize,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    text_compression_level=text_compression_level,
                    compute_mask=compute_mask,
                    **mask_args,
                )
            else:
                dataset_map = OrderedDict()
                self.dataset_map = {}
                multi_corpus_keys = [k.strip() for k in task_cfg.multi_corpus_keys.split(",")]
                corpus_idx_map = {k: idx for idx, k in enumerate(multi_corpus_keys)}
                data_keys = [k.split(":") for k in split.split(",")]

                multi_corpus_sampling_weights = [float(val.strip()) for val in task_cfg.multi_corpus_sampling_weights.split(",")]
                data_weights = []

                for key, file_name in data_keys:
                    
                    k = key.strip()
                    manifest_path = os.path.join(data_path, "{}.tsv".format(file_name.strip()))                

                    # TODO: Remove duplication of code from the if block above
                    dataset_map[k] = FileAudioDataset(
                        manifest_path=manifest_path,
                        sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                        max_sample_size=self.cfg.max_sample_size,
                        min_sample_size=self.cfg.min_sample_size,
                        pad=task_cfg.labels is not None or task_cfg.enable_padding,
                        normalize=task_cfg.normalize,
                        num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                        text_compression_level=text_compression_level,
                        compute_mask=compute_mask,
                        corpus_key=corpus_idx_map[k],
                        **mask_args,
                    )

                    data_weights.append(multi_corpus_sampling_weights[corpus_idx_map[k]])

                self.dataset_map[split] = dataset_map
                
                if len(dataset_map) == 1:
                    self.datasets[split] = list(dataset_map.values())[0]
                else:
                    self.datasets[split] = MultiCorpusDataset(dataset_map, distribution=data_weights, seed=0, sort_indices=True)

        if getattr(task_cfg, "subsample", 1) < 1:
            self.datasets[split] = SubsampleDataset(
                self.datasets[split],
                task_cfg.subsample,
                shuffle=True,
                seed=task_cfg.seed,
            )

        if self.cfg.tpu and task_cfg.inferred_w2v_config.mask_channel_prob == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            # if "w2v_args" in actualized_cfg:
            if hasattr(actualized_cfg, "w2v_args"):
                model_cfg.w2v_args = actualized_cfg.w2v_args

        return model

    def post_save(self, cp_path, num_updates):
        if self.cfg.post_save_script is not None:
            logger.info(f"launching {self.cfg.post_save_script}")
            import os.path as osp
            from fairseq.file_io import PathManager

            eval_cp_path = osp.join(
                osp.dirname(cp_path), f"checkpoint_eval_{num_updates}.pt"
            )

            print(cp_path, eval_cp_path, osp.dirname(cp_path))

            assert PathManager.copy(
                cp_path, eval_cp_path, overwrite=True
            ), f"Failed to copy {cp_path} to {eval_cp_path}"

            import subprocess
            import shlex

            subprocess.call(shlex.split(f"{self.cfg.post_save_script} {eval_cp_path}"))
