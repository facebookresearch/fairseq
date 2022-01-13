# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import contextlib
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, open_dict, OmegaConf

import numpy as np
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.dataclass import ChoiceEnum


logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class SentencePredictionConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )
    init_token: Optional[int] = field(
        default=None,
        metadata={"help": "add token at the beginning of each batch item"},
    )
    separator_token: Optional[int] = field(
        default=None,
        metadata={"help": "add separator token between inputs"},
    )
    no_shuffle: bool = field(
        default=False,
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    add_prev_output_tokens: bool = field(
        default=False,
        metadata={
            "help": "add prev_output_tokens to sample, used for encoder-decoder arch"
        },
    )
    max_positions: int = field(
        default=512,
        metadata={"help": "max tokens per example"},
    )

    regression_target: bool = II("criterion.regression_target")
    classification_head_name: str = II("criterion.classification_head_name")
    seed: int = II("common.seed")


@register_task("sentence_prediction", dataclass=SentencePredictionConfig)
class SentencePredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    def __init__(self, cfg, data_dictionary, label_dictionary):
        super().__init__(cfg)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            os.path.join(cfg.data, "input0", "dict.txt"),
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        if not cfg.regression_target:
            label_dict = cls.load_dictionary(
                os.path.join(cfg.data, "label", "dict.txt"),
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(cfg, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(key, split):
            return os.path.join(self.cfg.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            try:
                dataset = data_utils.load_indexed_dataset(
                    split_path,
                    dictionary,
                    combine=combine,
                )
            except Exception as e:
                if "StorageException: [404] Path not found" in str(e):
                    logger.warning(f"dataset {e} not found")
                    dataset = None
                else:
                    raise e
            return dataset

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )
        input1 = make_dataset("input1", self.source_dictionary)

        if self.cfg.init_token is not None:
            input0 = PrependTokenDataset(input0, self.cfg.init_token)

        if input1 is None:
            src_tokens = input0
        else:
            if self.cfg.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.cfg.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.max_positions(),
            self.cfg.seed,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if self.cfg.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset["net_input"].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        if not self.cfg.regression_target:
            label_dataset = make_dataset("label", self.label_dictionary)
            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.label_dictionary.eos(),
                        ),
                        offset=-self.label_dictionary.nspecial,
                    )
                )
        else:
            label_path = "{0}.label".format(get_path("label", split))
            if os.path.exists(label_path):

                def parse_regression_target(i, line):
                    values = line.split()
                    assert (
                        len(values) == self.cfg.num_classes
                    ), f'expected num_classes={self.cfg.num_classes} regression target values on line {i}, found: "{line}"'
                    return [float(x) for x in values]

                with open(label_path) as h:
                    dataset.update(
                        target=RawLabelDataset(
                            [
                                parse_regression_target(i, line.strip())
                                for i, line in enumerate(h.readlines())
                            ]
                        )
                    )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.cfg.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        model.register_classification_head(
            self.cfg.classification_head_name,
            num_classes=self.cfg.num_classes,
        )

        return model

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
