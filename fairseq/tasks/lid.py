# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import json
import os
import logging
from typing import Optional

from omegaconf import II

from fairseq import metrics, options, utils
from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    LanguagePairDataset,
    IndexedDataset,
    FairseqDataset,
    ConcatSentencesDataset,
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
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


@dataclass
class LIDConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sequence"},
    )

    seed: int = 3
    no_shuffle: bool = False


@register_task("lid", dataclass=LIDConfig)
class LIDTask(FairseqTask):
    """
        Language Identification Classification

        inspired by `SentencePredictionTask`
    """

    cfg: LIDConfig

    def __init__(self, cfg: LIDConfig, dictionary, label_dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary
        self._label_dictionary = label_dictionary

        logger.info(f"len(dictionary) = {len(dictionary)}")


    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # load dictionary
        dict_path = os.path.join(paths[0], "dict.sentence.txt")
        dictionary = cls.load_dictionary(dict_path)

        label_dict_path = os.path.join(paths[0], "dict.label.txt")
        label_dictionary = cls.load_dictionary(label_dict_path)

        return cls(args, dictionary, label_dictionary)

    def build_model(self, cfg: FairseqDataclass):
        from fairseq import models, quantization_utils

        model = models.build_model(cfg, self)
        model = quantization_utils.quantize_model_scalar(model, cfg)
        num_classes = len(self.label_dictionary)-self.label_dictionary.nspecial
        model.register_classification_head('sentence_classification_head', num_classes=num_classes)

        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != 'train':
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        def get_path(key, split):
            return os.path.join(data_path, f"{split}.{key}")

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            try:
                dataset = data_utils.load_indexed_dataset(
                    split_path,
                    self.dictionary,
                    self.cfg.dataset_impl,
                    combine=combine,
                )
            except Exception as e:
                if "StorageException: [404] Path not found" in str(e):
                    logger.warning(f"dataset {e} not found")
                    dataset = None
                else:
                    raise e
            return dataset

        input0 = make_dataset("sentence", self.source_dictionary)
        label_dataset = make_dataset("label", self.label_dictionary)
        src_tokens = input0

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

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_tokens))

        if self.cfg.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        self.datasets[split] = dataset
















