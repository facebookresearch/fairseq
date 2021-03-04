# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from dataclasses import dataclass, field
from argparse import Namespace
from omegaconf import II, MISSING

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.dataclass import FairseqDataclass


logger = logging.getLogger(__name__)


@dataclass
class SpeechToTextTaskConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={"help": "manifest root path"}
    )
    data_config_yaml: str = field(
        default="config.yaml",
        metadata={"help": "Configuration YAML filename (under manifest root)"}
    )
    max_source_positions: int = field(
        default=6000,
        metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the target sequence"}
    )

    # Inherit from other configs
    train_subset: str = II("dataset.train_subset")
    seed: int = II("common.seed")


@register_task("speech_to_text", dataclass=SpeechToTextTaskConfig)
class SpeechToTextTask(FairseqTask):

    def __init__(self, cfg, tgt_dict):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(op.join(cfg.data, cfg.data_config_yaml))

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        data_cfg = S2TDataConfig(op.join(cfg.data, cfg.data_config_yaml))
        dict_path = op.join(cfg.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): {len(tgt_dict)}"
        )
        if getattr(cfg, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(cfg, tgt_dict)

    def build_criterion(self, cfg):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and cfg.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "criterion.ignore_prefix_size=1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(cfg, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.cfg)
        bpe_tokenizer = self.build_bpe(self.cfg)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.cfg.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def build_generator(
        self,
        models,
        cfg,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and cfg.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, cfg, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, cfg):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, cfg):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
