# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    DenoisingDataset,
    Dictionary,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("denoising")
class DenoisingTask(LegacyFairseqTask):
    """
    Denoising task for applying sequence to sequence denoising. (ie. BART)
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments"
            " per sample for dataset",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete_doc",
            type=str,
            help="mode for breaking sentence",
        )
        parser.add_argument(
            "--mask",
            default=0.0,
            type=float,
            help="fraction of words/subwords that will be masked",
        )
        parser.add_argument(
            "--mask-random",
            default=0.0,
            type=float,
            help="instead of using [MASK], use random token this often",
        )
        parser.add_argument(
            "--insert",
            default=0.0,
            type=float,
            help="insert this percentage of additional random tokens",
        )
        parser.add_argument(
            "--permute",
            default=0.0,
            type=float,
            help="take this proportion of subwords and permute them",
        )
        parser.add_argument(
            "--rotate",
            default=0.5,
            type=float,
            help="rotate this proportion of inputs",
        )
        parser.add_argument(
            "--poisson-lambda",
            default=3.0,
            type=float,
            help="randomly shuffle sentences for this proportion of inputs",
        )
        parser.add_argument(
            "--permute-sentences",
            default=0.0,
            type=float,
            help="shuffle this proportion of sentences in all inputs",
        )
        parser.add_argument(
            "--mask-length",
            default="subword",
            type=str,
            choices=["subword", "word", "span-poisson"],
            help="mask length to choose",
        )
        parser.add_argument(
            "--replace-length",
            default=-1,
            type=int,
            help="when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = self.dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        dictionary = Dictionary.load(os.path.join(args.data, "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = StripTokenDataset(dataset, self.dictionary.eos())

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 2,  # one less for <s> and one for </s>
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            document_sep_len=0,
        )

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())

        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.args.mask_length != "subword"
            else None
        )

        self.datasets[split] = DenoisingDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.mask_idx,
            mask_whole_words,
            shuffle=self.args.shuffle_instance,
            seed=self.seed,
            args=self.args,
        )
        logger.info(
            "Split: {0}, Loaded {1} samples of denoising_dataset".format(
                split,
                len(self.datasets[split]),
            )
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary
