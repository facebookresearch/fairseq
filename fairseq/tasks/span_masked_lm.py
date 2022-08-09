# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from omegaconf import II, MISSING

from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.data.span_mask_tokens_dataset import SpanMaskedTokensDataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from ..data.indexed_dataset import get_available_dataset_impl

logger = logging.getLogger(__name__)

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class SpanMaskedLMConfig(FairseqDataclass):
    shuffle: bool = field(
        default=False,
    )
    noise_density: float = field(
        default=0.15,
        metadata={"help": "What fraction of the tokens to select as noise"},
    )
    mean_noise_span_length: float = field(
        default=3,
        metadata={"help": "Mean noise span length, must be >= 1"},
    )
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, "
            "will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    include_target_tokens: bool = field(
        default=False,
        metadata={
            "help": "include target tokens in model input. this is used for data2vec"
        },
    )


@register_task("span_masked_lm", dataclass=SpanMaskedLMConfig)
class SpanMaskedLMTask(FairseqTask):
    """
    Span masked language modeling task. (ie. T5)
    """

    cfg: SpanMaskedLMConfig

    def __init__(self, cfg, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, cfg: SpanMaskedLMConfig, **kwargs):
        """Setup the task."""
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(cfg, "shuffle"):
            cfg.shuffle = False
        return cls(cfg, dictionary)

    def _load_dataset_split(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.cfg.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = StripTokenDataset(dataset, self.dictionary.eos())

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample - 2,  # one less for <s> and one for </s>
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
            document_sep_len=0,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())
        return dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset = self._load_dataset_split(split, epoch, combine)

        self.datasets[split] = SpanMaskedTokensDataset(
            dataset,
            self.dictionary,
            noise_density=self.cfg.noise_density,
            mean_noise_span_length=self.cfg.mean_noise_span_length,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
        )
        logger.info(
            "Split: {0}, Loaded {1} samples of span_masked_tokens_dataset".format(
                split,
                len(self.datasets[split]),
            )
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We assume that the input begins with a
        bos symbol (`<s>`) and ends with an eos symbol (`</s>`).
        """
        pad = self.source_dictionary.pad()
        eos = self.source_dictionary.eos()
        src_dataset = TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=self.cfg.tokens_per_sample - 2,  # for <s> and </s>
            pad=pad,
            eos=eos,
            break_mode=self.cfg.sample_break_mode,
            document_sep_len=0,
        )
        prev_output_tokens = PrependTokenDataset(
            StripTokenDataset(src_dataset, eos), eos
        )
        src_dataset = PadDataset(src_dataset, pad_idx=pad, left_pad=False)
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                    "prev_output_tokens": PadDataset(
                        prev_output_tokens, pad_idx=pad, left_pad=False
                    ),
                },
                "target": src_dataset,
            },
            sizes=[np.array(src_lengths)],
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary
