# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import re

import numpy as np
import torch

from examples.speech_text_joint_to_text.data.pair_denoising_dataset import (
    LanguagePairDenoisingDataset,
)
from fairseq import utils
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    LanguagePairDataset,
    ResamplingDataset,
    TransformEosConcatLangPairDataset,
    TransformEosLangPairDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

logger = logging.getLogger(__name__)


def gen_whole_word_mask(args, dictionary):
    def is_beginning_of_word(i):
        if i < dictionary.nspecial:
            # special elements are always considered beginnings
            return True
        tok = dictionary[i]
        if tok.startswith("madeupword"):
            return True

        if tok in ["<unk>", "<s>", "</s>", "<pad>"]:
            return True
        return tok.startswith("\u2581")

    if args.use_mask_whole_words:
        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
    else:
        # it will mask every token as word leading token, since no bpe model is loaded for phoneme tokens
        return get_whole_word_mask(args, dictionary)
    return mask_whole_words


@register_task("paired_denoising")
class PairedDenoisingTask(TranslationTask):

    LANG_TAG_TEMPLATE = "<lang:{}>"  # Tag for language (target)

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        # bart setting
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
            "--poisson-lambda",
            default=3.0,
            type=float,
            help="randomly shuffle sentences for this proportion of inputs",
        )
        parser.add_argument(
            "--mask-length",
            default="span-poisson",
            type=str,
            choices=["subword", "word", "span-poisson"],
            help="mask length to choose",
        )
        parser.add_argument(
            "--replace-length",
            default=1,
            type=int,
            help="when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
        )

        # multi-lingual
        parser.add_argument(
            "--multilang-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for sample ratios across multiple datasets",
        )
        parser.add_argument(
            "--lang-pairs",
            default="",
            metavar="PAIRS",
            help="comma-separated list of language pairs (in training order): phnen-en,phnfr-fr,phnit-it. Do masking",
        )
        parser.add_argument(
            "--lang-pairs-bitext",
            default="",
            metavar="PAIRS",
            help="comma-separated list of language pairs (in training order): en-de,en-fr,de-fr. No masking",
        )
        parser.add_argument("--add-src-lang-token", default=False, action="store_true")
        parser.add_argument("--add-tgt-lang-token", default=False, action="store_true")
        parser.add_argument(
            "--no-whole-word-mask-langs",
            type=str,
            default="",
            metavar="N",
            help="languages without spacing between words dont support whole word masking",
        )
        parser.add_argument(
            "--use-mask-whole-words", default=False, action="store_true"
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        paths = args.data.split(":")
        assert len(paths) > 0
        src_dict = Dictionary.load(
            os.path.join(paths[0], "src_dict.txt")
        )  # assume all languages share a source dictionary
        tgt_dict = Dictionary.load(
            os.path.join(paths[0], "tgt_dict.txt")
        )  # assume all languages share a target dictionary

        lang_pairs = args.lang_pairs + "," + args.lang_pairs_bitext
        lang_pairs = re.sub(",$", "", re.sub("^,", "", lang_pairs))
        src_langs = [lp.split("-")[0] for lp in lang_pairs.split(",")]
        tgt_langs = [lp.split("-")[1] for lp in lang_pairs.split(",")]

        if args.add_src_lang_token:
            for lang in src_langs:
                assert (
                    src_dict.index(PairedDenoisingTask.LANG_TAG_TEMPLATE.format(lang))
                    != src_dict.unk()
                )
        if args.add_tgt_lang_token:
            for lang in tgt_langs:
                assert (
                    tgt_dict.index(PairedDenoisingTask.LANG_TAG_TEMPLATE.format(lang))
                    != tgt_dict.unk()
                )

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, src_dict, tgt_dict)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        # check mask token
        self.mask_idx = self.src_dict.index("<mask>")
        assert self.mask_idx != self.src_dict.unk()
        self.lang_pairs = args.lang_pairs
        self.lang_pairs_bitext = args.lang_pairs_bitext
        self.args = args

    @classmethod
    def language_pair_denoising_dataset(
        cls,
        data_path,
        do_mask,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        mask_idx,
        mask_whole_words,
        seed,
        args,
        dataset_impl,
        combine=False,
        left_pad_source=True,
        left_pad_target=False,
        max_source_positions=1024,
        max_target_positions=1024,
        shuffle=True,
        src_lang_id=None,
        tgt_lang_id=None,
    ):
        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(
                data_path, "{}.{}-{}.{}".format(split, src, tgt, lang)
            )
            return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

        src_datasets = []
        tgt_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else "")

            # infer langcode
            if split_exists(split_k, src, tgt, src, data_path):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            elif split_exists(split_k, tgt, src, src, data_path):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, data_path)
                    )

            src_dataset = data_utils.load_indexed_dataset(
                prefix + src, src_dict, dataset_impl
            )
            src_datasets.append(src_dataset)

            tgt_dataset = data_utils.load_indexed_dataset(
                prefix + tgt, tgt_dict, dataset_impl
            )
            if tgt_dataset is not None:
                tgt_datasets.append(tgt_dataset)

            logger.info(
                "{} {} {}-{} {} examples".format(
                    data_path, split_k, src, tgt, len(src_datasets[-1])
                )
            )

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
            tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        else:
            sample_ratios = [1] * len(src_datasets)
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            if len(tgt_datasets) > 0:
                tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            else:
                tgt_dataset = None

        eos = None

        tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
        if not do_mask:
            return LanguagePairDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                eos=eos,
                shuffle=shuffle,
                src_lang_id=src_lang_id,
                tgt_lang_id=tgt_lang_id,
            )

        return LanguagePairDenoisingDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            mask_idx,
            mask_whole_words,
            seed,
            args,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            eos=eos,
            shuffle=shuffle,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
        )

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def resample_datasets(self, lang_datasets, lang_pairs_all, epoch):
        # For train subset, additionally up or down sample languages.
        if self.args.multilang_sampling_alpha == 1.0:
            return lang_datasets

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        sample_probs = self._get_sample_prob(dataset_lengths)
        logger.info(
            "Sample probability by language pair: {}".format(
                {
                    lp: "{0:.4f}".format(sample_probs[id])
                    for id, lp in enumerate(lang_pairs_all)
                }
            )
        )
        size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
        logger.info(
            "Up/Down Sampling ratio by language: {}".format(
                {
                    lp: "{0:.2f}".format(size_ratio[id])
                    for id, lp in enumerate(lang_pairs_all)
                }
            )
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
        return resampled_lang_datasets

    def load_dataset_only(
        self, split, lang_pairs, do_mask=True, epoch=1, combine=False
    ):
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # TODO unk token will be considered as first word too, though it might be an unknown phoneme within a word
        # get_whole_word_mask returns a tensor (size V by 1 ) to indicate if a token is a word start token
        mask_whole_src_words = gen_whole_word_mask(self.args, self.src_dict)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(",")
        lang_datasets = []
        eos_bos = []
        lang_pairs = lang_pairs.split(",") if lang_pairs != "" else []
        assert len(lang_pairs) > 0
        for lp in lang_pairs:
            src, tgt = lp.split("-")
            lang_mask_whole_src_words = (
                mask_whole_src_words
                if src not in language_without_segmentations
                else None
            )

            end_token = (
                self.source_dictionary.index(
                    PairedDenoisingTask.LANG_TAG_TEMPLATE.format(src)
                )
                if self.args.add_src_lang_token
                else None
            )
            bos_token = (
                self.target_dictionary.index(
                    PairedDenoisingTask.LANG_TAG_TEMPLATE.format(tgt)
                )
                if self.args.add_tgt_lang_token
                else None
            )
            src_lang_id = None

            if self.args.add_src_lang_token or self.args.add_tgt_lang_token:
                eos_bos.append((end_token, bos_token))

            dataset = PairedDenoisingTask.language_pair_denoising_dataset(
                data_path,
                do_mask,
                split,
                src,
                self.source_dictionary,
                tgt,
                self.target_dictionary,
                self.mask_idx,
                lang_mask_whole_src_words,
                self.args.seed,
                self.args,
                self.args.dataset_impl,
                combine=combine,
                left_pad_source=utils.eval_bool(self.args.left_pad_source),
                left_pad_target=utils.eval_bool(self.args.left_pad_target),
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                src_lang_id=src_lang_id,
            )

            lang_datasets.append(dataset)

        if len(lang_datasets) == 0:
            return
        elif len(lang_datasets) == 1:
            dataset = lang_datasets[0]
            if self.args.add_src_lang_token or self.args.add_tgt_lang_token:
                end_token, bos_token = eos_bos[0]
                dataset = TransformEosLangPairDataset(
                    dataset,
                    src_eos=self.source_dictionary.eos(),
                    new_src_eos=end_token,
                    tgt_bos=self.target_dictionary.eos(),
                    new_tgt_bos=bos_token,
                )
        else:
            end_tokens = [item[0] for item in eos_bos if item[0] is not None]
            bos_tokens = [item[1] for item in eos_bos if item[1] is not None]
            lang_datasets = self.resample_datasets(lang_datasets, lang_pairs, epoch)
            dataset = TransformEosConcatLangPairDataset(
                lang_datasets,
                self.source_dictionary.eos(),
                self.target_dictionary.eos(),
                new_src_eos=end_tokens,
                new_tgt_bos=bos_tokens,
            )
        return dataset

    # split in (train, valid, test, ...)
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = self.load_dataset_only(
            split, self.lang_pairs, epoch=epoch, combine=combine
        )
