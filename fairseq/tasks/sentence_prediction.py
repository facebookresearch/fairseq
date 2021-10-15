# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from dataclasses import dataclass, field
from fairseq import utils
from typing import Optional
from argparse import Namespace
from omegaconf import II

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
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task



logger = logging.getLogger(__name__)

@dataclass
class SentencePredictionConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    num_classes: int = field(
        default=-1, metadata={"help": "number of classes or regression targets"},
    )
    init_token: Optional[int] = field(
        default=-1, metadata={"help": "add token at the beginning of each batch item"},
    )
    separator_token: Optional[int] = field(
        default=-1, metadata={"help": "add separator token between inputs"},
    )
    regression_target: bool = field(
        default=False,
    )
    no_shuffle: bool = field(
        default=False,
    )
    shorten_method: Optional[ChoiceEnum(["none", "truncate", "random_crop"])] = field(  # noqa: F821
        default="none", metadata={help: "if not none, shorten sequences that exceed --tokens-per-sample"},
    )
    shorten_data_split_list: Optional[str] = field(
        default="", metadata={help: "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'},
    )
    add_prev_output_tokens: Optional[bool] = field(
        default=False, metadata={help:"add prev_output_tokens to sample, used for encoder-decoder arch"},
    )
    tokens_per_sample: int = field(
        default=512,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False, metadata={"help": "pad to fixed length"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    max_positions: Optional[int] = II(
        "model.max_positions"
    )
    max_source_positions: Optional[int] = II(
        "model.max_source_positions"
    )

@register_task("sentence_prediction", dataclass=SentencePredictionConfig)
class SentencePredictionTask(LegacyFairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, "max_positions"):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        if not args.regression_target:
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, "label", "dict.txt"),
                source=False,
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(key, split):
            return os.path.join(self.args.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset
        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )
        input1 = make_dataset("input1", self.source_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.max_positions(),
            self.args.seed,
        )
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                    pad_length=self.args.tokens_per_sample if self.args.pad_to_fixed_length else None
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset["net_input"].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        if not self.args.regression_target:
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
                        len(values) == self.args.num_classes
                    ), f'expected num_classes={self.args.num_classes} regression target values on line {i}, found: "{line}"'
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

        if self.args.no_shuffle:
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

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)
        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_classes,
        )

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
