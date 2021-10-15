# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    RawLabelDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    encoders,
)
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("multilingual_masked_lm")
class MultiLingualMaskedLMTask(LegacyFairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--pad-to-fixed-length",
            default=False,
            action="store_true",
            help="pad examples to fixed length",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-whole-words",
            default=False,
            action="store_true",
            help="mask whole words; you may also want to set --bpe",
        )
        parser.add_argument(
            "--multilang-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for sample rations across multiple datasets",
        )
        parser.add_argument(
            "--lang-to-offline-shard-ratio",
            type=str,
            default="",
            help="absolute path of tsv file location to indicate lang to offline shard ratio.",
        )
        parser.add_argument(
            "--langs",
            type=str,
            help="list of langs",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def _get_whole_word_mask(self):
        # create masked input and targets
        if self.args.mask_whole_words:
            bpe = encoders.build_bpe(self.args)
            if bpe is not None:

                def is_beginning_of_word(i):
                    if i < self.source_dictionary.nspecial:
                        # special elements are always considered beginnings
                        return True
                    tok = self.source_dictionary[i]
                    if tok.startswith("madeupword"):
                        return True
                    try:
                        return bpe.is_beginning_of_word(tok)
                    except ValueError:
                        return True

                mask_whole_words = torch.ByteTensor(
                    list(map(is_beginning_of_word, range(len(self.source_dictionary))))
                )
        else:
            mask_whole_words = None
        return mask_whole_words

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    @staticmethod
    def _get_langs(args, epoch=1):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        languages = sorted(
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        )
        if args.langs:
            keep_langs = set(args.langs.split(","))
            languages = [lang for lang in languages if lang in keep_langs]
            assert len(languages) == len(keep_langs)

        return languages, data_path

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        languages, data_path = MultiLingualMaskedLMTask._get_langs(
            self.args, epoch
        )
        lang_to_offline_shard_ratio = None
        if self.args.lang_to_offline_shard_ratio != "":
            lang_to_offline_shard_ratio = {}
            assert os.path.exists(
                self.args.lang_to_offline_shard_ratio
            ), "provided offline shard ratio file doesn't exist: {0}".format(self.args.lang_to_offline_shard_ratio)
            with open(self.args.lang_to_offline_shard_ratio) as fin:
                for line in fin:
                    lang, ratio = line.strip().split('\t')
                    ratio = float(ratio)
                    lang_to_offline_shard_ratio[lang] = ratio
           
            logger.info(
                "Found offline sharded ratio: %s", lang_to_offline_shard_ratio,
            )

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: %s", {lang: id for id, lang in enumerate(languages)}
        )

        mask_whole_words = self._get_whole_word_mask()
        lang_datasets = []
        for lang_id, language in enumerate(languages):
            split_path = os.path.join(data_path, language, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode=self.args.sample_break_mode,
            )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

            src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
                dataset,
                self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
            )

            pad_length = None
            if self.args.pad_to_fixed_length:
                pad_length = self.args.tokens_per_sample

            lang_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                            pad_length=pad_length,
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "target": PadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                        pad_length=pad_length,
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                    "lang_id": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                },
                sizes=[src_dataset.sizes],
            )
            lang_datasets.append(lang_dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )        
        logger.info(
            "loaded total {} blocks for all languages".format(
                dataset_lengths.sum(),
            )
        )

        if split == self.args.train_subset:
            dataset_lengths_ratio_multiplier = np.ones(len(dataset_lengths))
            if lang_to_offline_shard_ratio is not None:            
                dataset_lengths_ratio_multiplier = []
                for lang in languages:
                    assert lang in lang_to_offline_shard_ratio, "Lang: {0} missing in offline shard ratio file: {1}".format(
                        lang, self.args.lang_to_offline_shard_ratio,
                    )
                    dataset_lengths_ratio_multiplier.append(lang_to_offline_shard_ratio[lang])
                dataset_lengths_ratio_multiplier = np.array(dataset_lengths_ratio_multiplier)
                true_dataset_lengths = dataset_lengths * dataset_lengths_ratio_multiplier
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(true_dataset_lengths)
            logger.info(
                "Sample probability by language: %s",
                {
                    lang: "{0:.4f}".format(sample_probs[id])
                    for id, lang in enumerate(languages)
                },
            )
            
            size_ratio = (sample_probs * true_dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: %s",
                {
                    lang: "{0:.4f}".format(size_ratio[id])
                    for id, lang in enumerate(languages)
                },
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(resampled_lang_datasets)
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + "_" + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            # [TODO]: This is hacky for now to print validation ppl for each
            # language individually. Maybe need task API changes to allow it
            # in more generic ways.
            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ",".join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
