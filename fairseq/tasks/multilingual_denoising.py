# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from omegaconf import II

from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    DenoisingDataset,
    Dictionary,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task

from .denoising import DenoisingConfig, DenoisingTask

logger = logging.getLogger(__name__)


@dataclass
class MultilingualDenoisingConfig(DenoisingConfig):
    multilang_sampling_alpha: float = field(
        default=1.0,
        metadata={"help": "smoothing alpha for sample ratios across multiple datasets"},
    )
    add_lang_token: bool = field(
        default=False,
        metadata={"help": ""},
    )
    langs: Optional[str] = field(
        default=None,
        metadata={"help": "language ids we are considering"},
    )
    no_whole_word_mask_langs: str = field(
        default="",
        metadata={
            "help": "languages without spacing between words don't support whole word masking"
        },
    )
    train_subset: str = II("common.train_subset")
    valid_subset: str = II("common.valid_subset")


@register_task("multilingual_denoising", dataclass=MultilingualDenoisingConfig)
class MultilingualDenoisingTask(DenoisingTask):

    cfg: MultilingualDenoisingConfig

    @classmethod
    def setup_task(cls, cfg: MultilingualDenoisingConfig, **kwargs):
        """Setup the task."""
        paths = cfg.data.split(":")
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))

        data_path = paths[0]
        if cfg.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = cfg.langs.split(",")

        if cfg.add_lang_token:
            for lang in languages:
                dictionary.add_symbol("[{}]".format(lang))

        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(cfg, "shuffle_instance"):
            cfg.shuffle_instance = False
        return cls(cfg, dictionary)

    def __init__(self, cfg: MultilingualDenoisingConfig, dictionary):
        super().__init__(cfg, dictionary)
        self.dictionary = dictionary

        # add mask token
        self.mask_idx = self.dictionary.add_symbol("<mask>")
        self.cfg = cfg

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling probability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob**self.cfg.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.cfg.data.split(":")
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        if self.cfg.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = self.cfg.langs.split(",")
            for name in languages:
                p = os.path.join(data_path, name)
                assert os.path.exists(p), "data not found: {}".format(p)

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: ", {lang: id for id, lang in enumerate(languages)}
        )

        mask_whole_words = get_whole_word_mask(self.cfg.bpe, self.dictionary)
        language_without_segmentations = self.cfg.no_whole_word_mask_langs.split(",")
        lang_datasets = []
        for language in languages:
            split_path = os.path.join(data_path, language, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.cfg.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            end_token = (
                self.source_dictionary.index("[{}]".format(language))
                if self.cfg.add_lang_token
                else self.source_dictionary.eos()
            )

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.cfg.tokens_per_sample - 2,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=end_token,
                break_mode=self.cfg.sample_break_mode,
            )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
            dataset = AppendTokenDataset(dataset, end_token)

            lang_mask_whole_words = (
                mask_whole_words
                if language not in language_without_segmentations
                else None
            )
            lang_dataset = DenoisingDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=self.cfg.shuffle_instance,
                seed=self.cfg.seed,
                mask=self.cfg.mask,
                mask_random=self.cfg.mask_random,
                insert=self.cfg.insert,
                rotate=self.cfg.rotate,
                permute_sentences=self.cfg.permute_sentences,
                bpe=self.cfg.bpe,
                replace_length=self.cfg.replace_length,
                mask_length=self.cfg.mask_length,
                poisson_lambda=self.cfg.poisson_lambda,
                eos=None
                if not self.cfg.add_lang_token
                else self.source_dictionary.index("[{}]".format(language)),
            )
            lang_datasets.append(lang_dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            "loaded total {} blocks for all languages".format(
                int(dataset_lengths.sum()),
            )
        )
        if split == self.cfg.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: {}".format(
                    {
                        lang: "{0:.4f}".format(sample_probs[id])
                        for id, lang in enumerate(languages)
                    }
                )
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: {}".format(
                    {
                        lang: "{0:.2f}".format(size_ratio[id])
                        for id, lang in enumerate(languages)
                    }
                )
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.cfg.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(
                resampled_lang_datasets,
            )
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + "_" + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            if split in self.cfg.valid_subset:
                self.cfg.valid_subset = self.cfg.valid_subset.replace(
                    split, ",".join(lang_splits)
                )

        with data_utils.numpy_seed(self.cfg.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )
