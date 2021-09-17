from argparse import Namespace
from fairseq.data.transform_eos_lang_pair_dataset import TransformEosLangPairDataset
from copy import deepcopy
from ..data.swav_dataset import load_langpair_weights_dataset

from fairseq.logging import metrics
import json
import logging
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from fairseq import options, utils
import fairseq
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    RawLabelDataset,
    ResamplingDataset,
    SortDataset,
    data_utils,
    encoders,
    AppendTokenDataset,
    LanguagePairDataset,
    StripTokenDataset,
    TruncateDataset,
    indexed_dataset,
)
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_masked_lm import MultiLingualMaskedLMTask
from fairseq.tasks.online_backtranslation import (
    OnlineBackTranslationTask,
    PiecewiseLinearFn,
)
from fairseq.data.nested_dictionary_dataset import _flatten as nested_dict_flatten
from fairseq.data.nested_dictionary_dataset import _unflatten as nested_dict_unflatten
from torch.utils.data.dataloader import default_collate
from collections import OrderedDict, defaultdict
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data.noising import NoisingDataset
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    load_langpair_dataset,
)
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.round_robin_zip_datasets import RoundRobinZipDatasets
import math
from fairseq.tasks import online_backtranslation
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from ..data.xlm_code_dictionary import XLMDictionary


"""
This defines a set of tasks for XLM codebase

"""

logger = logging.getLogger(__name__)


def xlm_lm_setup_task(cls, args, **kwargs):
    paths = utils.split_paths(args.data)
    assert len(paths) > 0
    # dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"), xlm_mode=True)
    dictionary = XLMDictionary.load(os.path.join(paths[0], "dict.txt"))
    logger.info("dictionary: {}: {} types".format(dictionary._class__.__name__, len(dictionary)))
    return cls(args, dictionary)


def xlm_load_dictionary(path, xlm_mode=True):
    # dictionary = Dictionary.load(path, xlm_mode=True)
    dictionary = XLMDictionary.load(path)
    logger.info("dictionary: {}: {} types".format(dictionary._class__.__name__, len(dictionary)))
    return dictionary


@register_task("multilingual_masked_lm_xlm")
class MultiLingualMaskedLMXLMTask(MultiLingualMaskedLMTask):
    """Main different from MultiLingualMaskedLMTask is
    replace bos with dictionary.eos()
    """

    @classmethod
    def setup_task(cls, args, **kwargs):
        return xlm_lm_setup_task(cls, args, **kwargs)

    @property
    def prepend_token(self):
        prepend_tok = self.source_dictionary.eos()
        return prepend_tok

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        languages = sorted(
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        )

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: {}".format(
                {lang: id for id, lang in enumerate(languages)}
            )
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
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.prepend_token)

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

            lang_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                        "src_langs": RawLabelDataset(
                            [lang_id] * src_dataset.sizes.shape[0]
                        ),
                    },
                    "target": PadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                    "lang_id": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                },
                sizes=[src_dataset.sizes],
            )
            lang_datasets.append(lang_dataset)

        self.build_multilingual_dataset(languages, lang_datasets, split, epoch)

    def build_multilingual_dataset(
        self, languages, lang_datasets, split, epoch, train_lang_sep=False
    ):
        """
        MultiLingualSwavLMTask sample regardless of languages, like MultiLingualMaskedLMTask
            - restricted version must sample same no. of data per language in a batch
        """
        dataset_lengths = np.array([len(d) for d in lang_datasets], dtype=float,)
        logger.info(
            "loaded total {} blocks for all languages".format(dataset_lengths.sum(),)
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: {}".format(
                    {
                        lang: "{0:.4f}".format(sample_probs[id])
                        for id, lang in enumerate(languages)
                    }
                ),
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
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(resampled_lang_datasets)
            if train_lang_sep:
                for l, d in zip(languages, lang_datasets):
                    self.datasets[f"{split}_{l}"] = d
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
            dataset, sort_order=[shuffle, dataset.sizes],
        )


class MassMaskExpandDataset(FairseqDataset):
    """
    This conduct span-based masking for MASS pretraining.
    Input: src_tokens: [a b c d e f g h i j k]
    Output: {
        src_tokens:         [a b c d _ _ _ _ i j k]
        prev_output_tokens: [d e f g ]
        target:             [e f g h ]
    }
    """

    def __init__(
        self,
        defn,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        sizes=None,
        seed: int = 1,
        span_len: int = 100000,
        word_mass: float = 0.5,
        no_input_noise=False,
    ) -> None:
        super().__init__()
        assert 0 < word_mass < 1
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        # NOTE target is decoder input, output is decoder output
        self.seed = seed
        self.span_len = span_len
        self.word_mass = word_mass
        self.no_input_noise = no_input_noise
        self.sizes = [sizes] if not isinstance(sizes, (list, tuple)) else sizes

        self.defn = nested_dict_flatten(defn)
        first = None
        for v in self.defn.values():
            if not isinstance(v, (FairseqDataset, torch.utils.data.Dataset)):
                raise ValueError("Expected Dataset but found: {}".format(v.__class__))
            first = first or v
            assert len(v) == 0 or len(v) == len(first), "dataset lengths must match"
        self._len = len(first)

        self.epoch = 0
        # FIXME nxphi: temporaily fox word_mask, word_keep, word_rand = 0.8, 0.1, 0.1
        self.word_mask = 0.8
        self.word_keep = 0.1
        self.word_rand = 0.1
        self.pred_probs = torch.FloatTensor(
            [self.word_mask, self.word_keep, self.word_rand]
        )
        self.foreign_flat_collater_map = {
            "net_input.prev_output_tokens": "net_input.src_tokens",
            "net_input.tgt_lengths": "net_input.src_lengths",
            "net_input.tgt_langs": "net_input.src_langs",
            "target": "net_input.src_tokens",
        }
        self.noise_input_map = {}
        if self.no_input_noise:
            self.noise_input_map = {
                "net_noise_input.src_tokens": "net_input.src_tokens",
                "net_noise_input.src_lengths": "net_input.src_lengths",
                "net_noise_input.src_langs": "net_input.src_langs",
                "net_noise_input.prev_output_tokens": "net_input.src_tokens",
                "net_noise_input.tgt_lengths": "net_input.src_lengths",
                "net_noise_input.tgt_langs": "net_input.src_langs",
            }

    def unfold_segments(self, segs):
        """Unfold the random mask segments, for example:
           The shuffle segment is [2, 0, 0, 2, 0],
           so the masked segment is like:
           [1, 1, 0, 0, 1, 1, 0]
           [1, 2, 3, 4, 5, 6, 7] (positions)
           (1 means this token will be masked, otherwise not)
           We return the position of the masked tokens like:
           [1, 2, 5, 6]
        """
        pos = []
        curr = 1  # We do not mask the start token
        for seg in segs:
            if seg >= 1:
                pos.extend([curr + i for i in range(seg)])
                curr += seg
            else:
                curr += 1
        return np.array(pos)

    def shuffle_segments(self, segs, unmasked_tokens):
        """
        We control 20% mask segment is at the start of sentences
                   20% mask segment is at the end   of sentences
                   60% mask segment is at random positions,
        """

        p = np.random.random()
        if p >= 0.8:
            shuf_segs = segs[1:] + unmasked_tokens
        elif p >= 0.6:
            shuf_segs = segs[:-1] + unmasked_tokens
        else:
            shuf_segs = segs + unmasked_tokens

        random.shuffle(shuf_segs)

        if p >= 0.8:
            shuf_segs = segs[0:1] + shuf_segs
        elif p >= 0.6:
            shuf_segs = shuf_segs + segs[-1:]
        return shuf_segs

    def get_segments(self, mask_len, span_len):
        segs = []
        while mask_len >= span_len:
            segs.append(span_len)
            mask_len -= span_len
        if mask_len != 0:
            segs.append(mask_len)
        return segs

    def mask_word(self, w):
        # old version from MASS
        _w_real = w
        _w_rand = np.random.randint(self.vocab_size, size=w.shape)
        _w_mask = np.full(w.shape, self.mask_idx)
        probs = torch.multinomial(self.pred_probs, len(_w_real), replacement=True)
        _w = (
            _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy()
        )
        return _w

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(s[index] for s in self.sizes)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if len(self.sizes) == 1:
            return self.sizes[0][index]
        else:
            return (s[index] for s in self.sizes)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        for ds in self.defn.values():
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch(indices)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(ds.can_reuse_epoch_itr_across_epochs for ds in self.defn.values())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.defn.values():
            ds.set_epoch(epoch)

    def __getitem__(self, idx):
        """
        expect in {
            "net_input": {
                "src_tokens": PadDataset(), # default src_tokens, pre-mass
                "src_lengths": NumelDataset(src_dataset, reduce=False),
                "src_langs": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
            },
            "target": PadDataset(
                out_dataset,
                pad_idx=self.target_dictionary.pad(),
                left_pad=False,
            ),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
            "lang_id": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
        }
        add to net_input:
            "prev_output_tokens": PadDataset(),
            "tgt_lengths": NumelDataset(tgt_dataset, reduce=False),
            "tgt_langs": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
        add to dict:
            "target": PadDataset(
                out_dataset,
                pad_idx=self.target_dictionary.pad(),
                left_pad=False,
            ),
        """
        dict_item = OrderedDict((k, ds[idx]) for k, ds in self.defn.items())
        assert isinstance(dict_item, (dict, OrderedDict)), f"type: {type(dict_item)}"
        assert "net_input.src_tokens" in dict_item, f"{dict_item.keys()}"
        src_tokens = dict_item["net_input.src_tokens"]
        # NOTE: must clone() as CPU tensor and numpy share same memory and src_tokens is modified
        _src_tokens = src_tokens.clone()
        sz = len(src_tokens)
        mask_len = round(sz * self.word_mass)
        unmasked_tokens = [0] * (sz - mask_len - 1)
        segs = self.get_segments(mask_len, self.span_len)

        assert (
            self.mask_idx not in src_tokens
        ), "Dataset contains mask_idx (={}), this is not expected!".format(
            self.mask_idx,
        )
        words = src_tokens.numpy()
        shuf_segs = self.shuffle_segments(segs, unmasked_tokens)
        pos_i = self.unfold_segments(shuf_segs)
        output_i = words[pos_i].copy()
        target_i = words[pos_i - 1].copy()
        words[pos_i] = self.mask_word(words[pos_i])

        # src_lengths doesn't change!
        prev_output_tokens = torch.from_numpy(target_i)
        output = torch.from_numpy(output_i)
        dict_item["net_input.src_tokens"] = torch.from_numpy(words)
        # foreign item
        dict_item["net_input.prev_output_tokens"] = prev_output_tokens
        dict_item["net_input.tgt_lengths"] = torch.numel(prev_output_tokens)
        dict_item["net_input.tgt_langs"] = dict_item["net_input.src_langs"]
        dict_item["target"] = output
        if self.no_input_noise:
            # src_lengths and src_langs does not change
            for k in self.noise_input_map.keys():
                if k.startswith("net_noise_input"):
                    dict_item[k] = dict_item[k.replace("net_noise_input", "net_input")]
            dict_item["net_input.src_tokens"] = _src_tokens
        return dict_item

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        for k, ds in self.defn.items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except NotImplementedError:
                sample[k] = default_collate([s[k] for s in samples])
        # collate foreign item
        for _map in [self.foreign_flat_collater_map, self.noise_input_map]:
            for k, v in _map.items():
                try:
                    sample[k] = self.defn[v].collater([s[k] for s in samples])
                except NotImplementedError:
                    sample[k] = default_collate([s[k] for s in samples])
        sample = nested_dict_unflatten(sample)
        return sample


@register_task("multilingual_mass_xlm")
class MultilingualMassXLMTask(MultiLingualMaskedLMTask):
    """
    MASS pretraining for XLM codebase models
    """

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return xlm_lm_setup_task(cls, args, **kwargs)

    @staticmethod
    def add_args(parser):
        MultiLingualMaskedLMTask.add_args(parser=parser)
        MultilingualMassXLMTask.add_mass_args(parser=parser)

    @classmethod
    def add_mass_args(cls, parser):
        parser.add_argument(
            "--span-len", type=int, default=10000, help="Span coefficient"
        )
        parser.add_argument(
            "--word-mass",
            type=float,
            default=0.5,
            help="Randomly mask input words (0 to disable)",
        )

    def max_positions(self):
        return self.args.tokens_per_sample

    def build_multilingual_dataset(
        self, languages, lang_datasets, split, epoch, train_lang_sep=False
    ):
        """
        MultiLingualSwavLMTask sample regardless of languages, like MultiLingualMaskedLMTask
            - restricted version must sample same no. of data per language in a batch
        """
        dataset_lengths = np.array([len(d) for d in lang_datasets], dtype=float,)
        logger.info(
            "loaded total {} blocks for all languages".format(dataset_lengths.sum(),)
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: {}".format(
                    {
                        lang: "{0:.4f}".format(sample_probs[id])
                        for id, lang in enumerate(languages)
                    }
                ),
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
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(resampled_lang_datasets)
            if train_lang_sep:
                for l, d in zip(languages, lang_datasets):
                    self.datasets[f"{split}_{l}"] = d
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
            dataset, sort_order=[shuffle, dataset.sizes],
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        languages = sorted(
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        )

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: ", {lang: id for id, lang in enumerate(languages)}
        )

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
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            # NOTE: for xlm, prepend eos not, bos
            dataset = PrependTokenDataset(dataset, self.source_dictionary.eos())

            lang_dataset = MassMaskExpandDataset(
                {
                    "net_input": {
                        "src_tokens": PadDataset(
                            dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(dataset, reduce=False),
                        "src_langs": RawLabelDataset(
                            [lang_id] * dataset.sizes.shape[0]
                        ),
                    },
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(dataset, reduce=True),
                    "lang_id": RawLabelDataset([lang_id] * dataset.sizes.shape[0]),
                },
                vocab=self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                sizes=[dataset.sizes],
                seed=self.args.seed,
                span_len=self.args.span_len,
                word_mass=self.args.word_mass,
            )

            lang_datasets.append(lang_dataset)

        self.build_multilingual_dataset(languages, lang_datasets, split, epoch)


# NOTE -- XLM Translation tasks


def load_xlm_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
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
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
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
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        # FIXME nxphi: don't know why enabling src_lang_id this cause
    )


def xlm_setup_translation_task(cls, cfg, **kwargs):
    paths = utils.split_paths(cfg.data)
    assert len(paths) > 0
    # find language pair automatically
    if cfg.source_lang is None or cfg.target_lang is None:
        cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
    if cfg.source_lang is None or cfg.target_lang is None:
        raise Exception("Could not infer language pair, please provide it explicitly")

    # load dictionaries
    src_dict = xlm_load_dictionary(
        os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
    )
    tgt_dict = xlm_load_dictionary(
        os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
    )
    assert src_dict.pad() == tgt_dict.pad()
    assert src_dict.eos() == tgt_dict.eos()
    assert src_dict.unk() == tgt_dict.unk()
    logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
    logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

    return cls(cfg, src_dict, tgt_dict)


@register_task("translation_xlm_debug", dataclass=TranslationConfig)
class TranslationDebugXLMTask(TranslationTask):
    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        logger.warning(f"setup xlm task....")
        return xlm_setup_translation_task(cls, cfg, **kwargs)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_xlm_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )


@register_task("online_backtranslation_xlm")
class OnlineBackTranslationXLMTask(OnlineBackTranslationTask):
    """
    For original backtranslation
        1. lang-specific tokens are prepended __{lang}__ like mBART (but not for mBART)
            like _lang_token_index
        2. in build_model, `add_secial_tokens_to_dict_and_model` adds lang-specifier to embeddings
            # <mask> is first added should reuse the pretrained model dict which already has <mask>
            # dictionary.add_symbol(lang_token): f"__{lang}__"
    For XLM backtranslation, we need to:
        0. replace Dictionary with XLMDictionary
        1. ensure args.left_pad_source=False
        1. remodify XLMTransformerModel so that:
            1.1 mono_langs is pass and build
            1.2 first token in dictionary.index(<mono_langs>)
            1.3 replace first_token = eos and build src_langs from first_token
        2. ?
    Expected args for XLM
        --mono-langs ???
        --n_langs size of ${mono_langs} !!!!!
        --valid-lang-pairs en-ro ?
    For MASS:
        --lambda-dae 0
    """

    def __init__(self, args, common_dict, mono_langs, valid_lang_pairs):
        super().__init__(args, common_dict, mono_langs, valid_lang_pairs)
        self.common_dict_2nd = deepcopy(common_dict)

    @staticmethod
    def add_args(parser):
        OnlineBackTranslationTask.add_args(parser)
        parser.add_argument(
            "--bt-train",
            default=False,
            action="store_true",
            help="Using train mode during bt process",
        )

    @classmethod
    def load_dictionary(cls, filename, xlm_mode=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return xlm_load_dictionary(filename)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        assert (
            not args.left_pad_source
        ), f"For XLM models, left_pad_source must be false"
        assert (
            not args.left_pad_target
        ), f"For XLM models, left_pad_target must be false"
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        assert args.mono_langs is not None

        mono_langs = args.mono_langs.split(",")
        valid_lang_pairs = args.valid_lang_pairs.split(",")

        # load dictionary
        dict_path = os.path.join(paths[0], "dict.txt")
        common_dict = cls.load_dictionary(dict_path, xlm_mode=True)

        return cls(args, common_dict, mono_langs, valid_lang_pairs)

    def build_model(self, args):
        model = super(TranslationTask, self).build_model(args)

        xlm_add_special_tokens_to_dict_and_model(
            self.common_dict, model, self.mono_langs
        )

        self.sequence_generators = {}
        for mono_lang in self.mono_langs:
            self.sequence_generators[mono_lang] = SequenceGenerator(
                [model],
                tgt_dict=self.dictionary,
                beam_size=1,
                max_len_a=1.3,
                max_len_b=5,
                min_len=5,
                # keep 1 to be able to prepend bos
                max_len=model.max_decoder_positions() - 1,
            )

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.bleu_sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        data = []
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))
            # REMOVE DENOISING AUTO ENCODER FOR MASS
            if len(self.lambda_dae.pieces) >= 1 and self.lambda_dae.pieces[0][1] > 0:
                data.append(
                    (f"{lang}-DENOISE", self.load_denoise_dataset(train_path, lang))
                )
            else:
                logger.warning(
                    f"Not building {lang}-/DENOISE because {self.args.lambda_dae=}"
                )

        return RoundRobinZipDatasets(OrderedDict(data))

    def display_samples_once_in_a_while(self, smp, mono_lang, other_lang):
        if 1 < self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr += 1
            return
        elif self._show_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr = 0
        else:
            self._show_samples_ctr += 1

        ln = smp["net_input"]["src_tokens"].shape[0]

        logger.info(
            f"(r:{self.args.distributed_rank}) : "
            f"{other_lang} ---> {mono_lang} "
            f"({other_lang} was generated by back-translation.) {ln} samples"
        )
        # bpe_symbol = "sentencepiece"
        # FIXME nxphi: need to change this to configureable
        bpe_symbol = "subword_nmt"
        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]

            src_str = self.dictionary.string(src_tokens, bpe_symbol)
            tgt_str = self.dictionary.string(tgt_tokens, bpe_symbol)
            logger.info(
                f"\n{i}\t\t[{other_lang} generated]  {src_str}\n"
                f"\t\t[{mono_lang} original ]  {tgt_str}\n"
                f"\t\t[ src tokens]  {src_tokens}\n"
            )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        dataset_keys = self.datasets["train"].datasets.keys()

        weights = {
            "BT": self.lambda_bt(update_num),
            "DENOISE": self.lambda_dae(update_num),
        }
        log_keys = {"BT": "bt_", "DENOISE": "dae_"}

        for dataset_key in dataset_keys:
            assert (
                dataset_key in sample or ignore_grad
            ), f"keymissing and not ignore[{ignore_grad=}]: {dataset_keys=}? {sample.keys()=}"
            if dataset_key not in sample and ignore_grad:
                # dummy batch found?
                assert all(
                    x in sample
                    for x in ["id", "nsentences", "ntokens", "net_input", "target"]
                ), f"{sample.keys()=}"
                smp = sample
                logger.warning(
                    f'invalid dummy batch found?: {sample["net_input"].keys()=}: sample: {sample}'
                )
            else:
                smp = sample[dataset_key]
            mono_lang, task_subtype = dataset_key.split("-")
            if weights[task_subtype] == 0:
                continue

            # logger.warning(f'train {dataset_key}...')
            if task_subtype == "BT":
                with torch.autograd.profiler.record_function("backtranslation"):
                    model.train(mode=self.args.bt_train)
                    # TODO: Could we translate to several language at once ?
                    # this would allow to share encoder_out and maximize GPU usage.
                    other_lang = self.get_other_lang(mono_lang)
                    self.backtranslate_sample(smp, mono_lang, other_lang)
                    self.display_samples_once_in_a_while(smp, mono_lang, other_lang)
                    model.train()

            # Like in FairseqTask.train_step
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, smp)
            loss *= weights[task_subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)

            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[task_subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(
        self,
        generator,
        models,
        sample,
        prefix_tokens=None,
        constraints=None,
        bos_token=None,
    ):
        if bos_token is None:
            # during inference
            # rely on valid-lang-pairs
            valid_lang_pair = self.valid_lang_pairs[0]
            src, tgt = valid_lang_pair.split("-")
            bos_token = online_backtranslation._lang_token_index(self.dictionary, tgt)
        assert bos_token is not None
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=bos_token,
            )


@register_task("augpara_online_backtranslation_xlm")
class AugParaOnlineBackTranslationXLMTask(OnlineBackTranslationXLMTask):
    """
    Integrate augmentation mined dataset into online BT
    """

    def __init__(self, args, common_dict, mono_langs, valid_lang_pairs):
        super().__init__(args, common_dict, mono_langs, valid_lang_pairs)
        self.lambda_augpara = PiecewiseLinearFn.from_string(args.lambda_augpara)

    @staticmethod
    def add_args(parser):
        OnlineBackTranslationXLMTask.add_args(parser)
        parser.add_argument(
            "--augpara-path", type=str, help="path to augmentation data"
        )
        parser.add_argument(
            "--augpara-pairs", type=str, help="pairs src-tgt of the augmentation data"
        )
        parser.add_argument(
            "--augpara-reverse",
            default=False,
            action="store_true",
            help="reverse each augpara data tgt->src",
        )
        parser.add_argument(
            "--lambda-augpara",
            default="1.0",
            type=str,
            metavar="N",
            help="augmentation data weight",
        )

    def load_translation_dataset(
        self, split: str, data_path: str, combine: bool = False, lang_pair=None
    ):
        # only judging with one language pair for the moment,
        # since ConcatDataset doesn't work as expected
        def build_dataset(_src, _tgt):
            # use the same function than TranslationTask
            src_tgt_dt = load_langpair_dataset(
                data_path,
                split,
                _src,
                self.common_dict,
                _tgt,
                self.common_dict,
                combine=combine,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                load_alignments=self.args.load_alignments,
                truncate_source=self.args.truncate_source,
                num_buckets=self.args.num_batch_buckets,
                shuffle=(split != "test"),
                prepend_bos_src=online_backtranslation._lang_token_index(
                    self.dictionary, _src
                ),
            )

            src_tgt_eos_dt = self._prepend_lang_bos_to_target(src_tgt_dt, _tgt)
            src_tgt_eos_dt.args = self.args
            return src_tgt_eos_dt

        if split == "train":
            assert lang_pair is not None
            src, tgt = lang_pair.split("-")
            return build_dataset(src, tgt)
        else:
            assert split in ["valid", "test"]
            datasets = []
            for i, pair in enumerate(self.valid_lang_pairs):
                src, tgt = pair.split("-")
                dataset = build_dataset(src, tgt)
                datasets.append((f"{src}{tgt}", dataset))
            return datasets

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        data = []
        args = self.args
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))
            # REMOVE DENOISING AUTO ENCODER FOR MASS
            if len(self.lambda_dae.pieces) >= 1 and self.lambda_dae.pieces[0][1] > 0:
                data.append(
                    (f"{lang}-DENOISE", self.load_denoise_dataset(train_path, lang))
                )
            else:
                logger.info(
                    f"Not building {lang}-/DENOISE because {self.args.lambda_dae=}"
                )
            # aug data
        augpara_paths = args.augpara_path.split(",")
        augpara_pairs = args.augpara_pairs.split(",")
        assert len(augpara_paths) == len(
            augpara_pairs
        ), f"{len(augpara_paths)=} != {len(augpara_pairs)}"
        for i, (p_path, p_pair) in enumerate(zip(augpara_paths, augpara_pairs)):
            # aug_path = os.path.join(p_path, f'train.{p_pair}')
            aug_path = p_path
            src, tgt = p_pair.split("-")
            logger.info(f"Loading aug data: {p_pair} at {aug_path}")
            dataset = self.load_translation_dataset("train", aug_path, lang_pair=p_pair)
            data.append((f"{src}{tgt}-AUG", dataset))
            if args.augpara_reverse:
                logger.info(f"Reversing aug data: {p_pair} at {aug_path}")
                r_dataset = self.load_translation_dataset(
                    "train", aug_path, lang_pair=f"{tgt}-{src}"
                )
                data.append((f"{tgt}{src}-AUG", r_dataset))

        return RoundRobinZipDatasets(OrderedDict(data))

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        dataset_keys = self.datasets["train"].datasets.keys()

        weights = {
            "BT": self.lambda_bt(update_num),
            "DENOISE": self.lambda_dae(update_num),
            "AUG": self.lambda_augpara(update_num),
        }
        log_keys = {"BT": "bt_", "DENOISE": "dae_", "AUG": "aug_"}

        for dataset_key in dataset_keys:
            smp = sample[dataset_key]
            mono_lang, task_subtype = dataset_key.split("-")
            if weights[task_subtype] == 0:
                continue

            if task_subtype == "BT":
                with torch.autograd.profiler.record_function("backtranslation"):
                    model.train(mode=self.args.bt_train)
                    # TODO: Could we translate to several language at once ?
                    # this would allow to share encoder_out and maximize GPU usage.
                    other_lang = self.get_other_lang(mono_lang)
                    self.backtranslate_sample(smp, mono_lang, other_lang)
                    self.display_samples_once_in_a_while(smp, mono_lang, other_lang)
                    model.train()

            # Like in FairseqTask.train_step
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, smp)
            loss *= weights[task_subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)

            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[task_subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        aug_sample_size = sum(x.get("aug_sample_size", 0) for x in logging_outputs)
        if aug_sample_size:
            aug_loss_sum = sum(x.get("aug_loss", 0) for x in logging_outputs)
            aug_loss_sum *= 1 / aug_sample_size / math.log(2)
            metrics.log_scalar("aug_loss", aug_loss_sum, aug_sample_size, round=3)

            aug_nll_loss_sum = sum(x.get("aug_nll_loss", 0) for x in logging_outputs)
            aug_ntokens = sum(x.get("aug_ntokens", 0) for x in logging_outputs)
            aug_nll_loss_sum *= 1 / aug_ntokens / math.log(2)
            metrics.log_scalar("aug_nll_loss", aug_nll_loss_sum, aug_ntokens, round=3)
            metrics.log_derived(
                "aug_ppl",
                lambda meters: utils.get_perplexity(meters["aug_nll_loss"].avg),
            )


@register_task("augpara_score_online_backtranslation_xlm")
class AugParaScoreOnlineBackTranslationXLMTask(AugParaOnlineBackTranslationXLMTask):
    """
        Integrate augmentation mined dataset into online BT with
            Aignment scores -> rank-based XE weighted cross entropy

    """

    def __init__(self, args, common_dict, mono_langs, valid_lang_pairs):
        super().__init__(args, common_dict, mono_langs, valid_lang_pairs)

    def scores_to_weights(self, scores):
        scores2weights = self.args.scores2weights
        params = [float(x) for x in self.args.scores2weights_params.split(",")]
        logger.info(f"scores2weights params: {params}")
        if scores2weights == "scale_min_max":
            logger.warning(
                f"WARNING: scale_min_max for positive similarity correlation (higher more similar), like cosine_sim, "
                f"for distance score, use --scores2weights neg_scale_min_max"
            )
            _min = params[0] if len(params) >= 1 else 0.0
            _max = params[1] if len(params) >= 2 else 1.0
            scores = np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min()) * (
                _max - _min
            ) + _min
        elif scores2weights == "neg_scale_min_max":
            logger.warning(
                f"WARNING: neg_scale_min_max for negative similarity correlation (higher more similar), like distances, "
                f"for cosine_sim score, use --scores2weights scale_min_max"
            )
            scores = -np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min())
        elif scores2weights == "scale_min_max_old":
            scores = np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min())
        elif scores2weights == "ones":
            weights = np.ones(shape=(len(scores,)))
        else:
            raise ValueError(f"{scores2weights} invalid")
        return weights

    @staticmethod
    def add_args(parser):
        AugParaOnlineBackTranslationXLMTask.add_args(parser)
        parser.add_argument(
            "--scores2weights",
            type=str,
            default="scale_min_max",
            help="path to augmentation data",
        )
        parser.add_argument(
            "--scores2weights-params",
            type=str,
            default="0,1",
            help="params for scores2weights",
        )
        parser.add_argument(
            "--no-use-weights",
            default=False,
            action="store_true",
            help="not using the weights",
        )

    def load_translation_dataset(
        self,
        split: str,
        data_path: str,
        combine: bool = False,
        lang_pair=None,
        pair_score=False,
    ):
        # only judging with one language pair for the moment,
        # since ConcatDataset doesn't work as expected
        def build_dataset(_src, _tgt):
            # use the  same function than TranslationTask
            if pair_score and self.args.no_use_weights:
                logger.info(f"Pair-Score: NO USE AUG_DATA WEIGHTS")
            if pair_score and not self.args.no_use_weights:
                pair_score_path = os.path.join(data_path, "index.pth")
                assert os.path.exists(pair_score_path), f"{pair_score_path} not found."
                logger.info(
                    f"Load aug data index {pair_score_path}, {self.args.scores2weights=}"
                )
                pth = torch.load(pair_score_path)
                scores = pth["scores"]
                weights = self.scores_to_weights(scores)

                src_tgt_dt = load_langpair_weights_dataset(
                    data_path=data_path,
                    split=split,
                    weights=weights,
                    src=_src,
                    src_dict=self.common_dict,
                    tgt=_tgt,
                    tgt_dict=self.common_dict,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    load_alignments=self.args.load_alignments,
                    truncate_source=self.args.truncate_source,
                    num_buckets=self.args.num_batch_buckets,
                    shuffle=(split != "test"),
                    prepend_bos_src=online_backtranslation._lang_token_index(
                        self.dictionary, _src
                    ),
                )
            else:
                src_tgt_dt = load_langpair_dataset(
                    data_path=data_path,
                    split=split,
                    src=_src,
                    src_dict=self.common_dict,
                    tgt=_tgt,
                    tgt_dict=self.common_dict,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    load_alignments=self.args.load_alignments,
                    truncate_source=self.args.truncate_source,
                    num_buckets=self.args.num_batch_buckets,
                    shuffle=(split != "test"),
                    prepend_bos_src=online_backtranslation._lang_token_index(
                        self.dictionary, _src
                    ),
                )

            src_tgt_eos_dt = self._prepend_lang_bos_to_target(src_tgt_dt, _tgt)
            src_tgt_eos_dt.args = self.args
            return src_tgt_eos_dt

        if split == "train":
            assert lang_pair is not None
            src, tgt = lang_pair.split("-")
            return build_dataset(src, tgt)
        else:
            assert split in ["valid", "test"]
            datasets = []
            for i, pair in enumerate(self.valid_lang_pairs):
                src, tgt = pair.split("-")
                dataset = build_dataset(src, tgt)
                datasets.append((f"{src}{tgt}", dataset))
            return datasets

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        data = []
        args = self.args
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))
            # REMOVE DENOISING AUTO ENCODER FOR MASS
            if len(self.lambda_dae.pieces) >= 1 and self.lambda_dae.pieces[0][1] > 0:
                data.append(
                    (f"{lang}-DENOISE", self.load_denoise_dataset(train_path, lang))
                )
            else:
                logger.info(
                    f"Not building {lang}-DENOISE because {self.args.lambda_dae=}"
                )
            # aug data
        augpara_paths = args.augpara_path.split(",")
        augpara_pairs = args.augpara_pairs.split(",")
        assert len(augpara_paths) == len(
            augpara_pairs
        ), f"{len(augpara_paths)=} != {len(augpara_pairs)}"
        for i, (p_path, p_pair) in enumerate(zip(augpara_paths, augpara_pairs)):
            # aug_path = os.path.join(p_path, f'train.{p_pair}')
            aug_path = p_path
            src, tgt = p_pair.split("-")
            logger.info(f"Loading aug data: {p_pair} at {aug_path}")
            dataset = self.load_translation_dataset(
                "train", aug_path, lang_pair=p_pair, pair_score=True
            )
            data.append((f"{src}{tgt}-AUG", dataset))
            if args.augpara_reverse:
                logger.info(f"Reversing aug data: {p_pair} at {aug_path}")
                r_dataset = self.load_translation_dataset(
                    "train", aug_path, lang_pair=f"{tgt}-{src}", pair_score=True
                )
                data.append((f"{tgt}{src}-AUG", r_dataset))

        return RoundRobinZipDatasets(OrderedDict(data))


def xlm_add_special_tokens_to_dict_and_model(
    dictionary: "fairseq.data.Dictionary",
    model: nn.Module,
    mono_langs: Sequence[str],
    extend_emb: bool = True,
) -> None:
    embs = model.encoder.embeddings
    vocab_size, embedding_dim = embs.weight.shape

    # The model may or may not have a '<mask>' embedding yet
    assert (
        len(dictionary) <= vocab_size <= len(dictionary) + 1
    ), f"Dictionary len ({len(dictionary)}) doesn't match embs shape ({embs.weight.shape})"
    # TODO: we should reuse the pretrained model dict which already has <mask>
    dictionary.add_symbol("<mask>")

    for lang in mono_langs:
        lang_token = online_backtranslation._lang_token(lang)
        dictionary.add_symbol(lang_token)
    logger.info(
        f"dictionary: {vocab_size} -> {len(dictionary)} tokens after adding {len(mono_langs)} lang tokens: {mono_langs}"
    )

    if len(dictionary) <= vocab_size:
        return

    if extend_emb:
        online_backtranslation.extend_embedding(embs, len(dictionary), dictionary.bos())
        if hasattr(model.encoder, "pred_layer"):
            enc_pred_proj_layer = model.encoder.pred_layer.proj
            online_backtranslation.extend_embedding(
                enc_pred_proj_layer, len(dictionary), dictionary.bos()
            )

        dec_embs = model.decoder.embeddings
        online_backtranslation.extend_embedding(
            dec_embs, len(dictionary), dictionary.bos()
        )
        pred_proj_layer = model.decoder.pred_layer.proj
        online_backtranslation.extend_embedding(
            pred_proj_layer, len(dictionary), dictionary.bos()
        )

    try:
        model.set_mono_langs(mono_langs)
    except Exception:
        logger.warning(
            f"Model {model.__class__.__name__} does not have set_mono_langs fn"
        )

    model.maybe_build_mono2langs_map(dictionary)


@register_task("online2_backtranslation_xlm")
class Online2ndBackTranslationXLMTask(OnlineBackTranslationTask):
    """
    For original backtranslation
        1. lang-specific tokens are prepended __{lang}__ like mBART (but not for mBART)
            like _lang_token_index
        2. in build_model, `add_secial_tokens_to_dict_and_model` adds lang-specifier to embeddings
            # <mask> is first added should reuse the pretrained model dict which already has <mask>
            # dictionary.add_symbol(lang_token): f"__{lang}__"
    For XLM backtranslation, we need to:
        0. replace Dictionary with XLMDictionary
        1. ensure args.left_pad_source=False
        1. remodify XLMTransformerModel so that:
            1.1 mono_langs is pass and build
            1.2 first token in dictionary.index(<mono_langs>)
            1.3 replace first_token = eos and build src_langs from first_token
        2. ?
    Expected args for XLM
        --mono-langs ???
        --n_langs size of ${mono_langs} !!!!!
        --valid-lang-pairs en-ro ?
        --eval-tokenized-bleu
        --eval-bleu-remove-bpe
        --eval-bleu-args '{"beam": 4, "lenpen": 0.6}'
    For MASS:
        --lambda-dae 0
    """

    def __init__(self, args, common_dict, mono_langs, valid_lang_pairs):
        super().__init__(args, common_dict, mono_langs, valid_lang_pairs)
        self.common_dict_2nd = deepcopy(common_dict)

    @staticmethod
    def add_args(parser):
        OnlineBackTranslationTask.add_args(parser)
        parser.add_argument(
            "--bt-train",
            default=False,
            action="store_true",
            help="Using train mode during bt process",
        )

    @classmethod
    def load_dictionary(cls, filename, xlm_mode=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return xlm_load_dictionary(filename, xlm_mode=xlm_mode)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        assert (
            not args.left_pad_source
        ), f"For XLM models, left_pad_source must be false"
        assert (
            not args.left_pad_target
        ), f"For XLM models, left_pad_target must be false"
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        assert args.mono_langs is not None

        mono_langs = args.mono_langs.split(",")
        valid_lang_pairs = args.valid_lang_pairs.split(",")

        # load dictionary
        dict_path = os.path.join(paths[0], "dict.txt")
        common_dict = cls.load_dictionary(dict_path, xlm_mode=True)

        return cls(args, common_dict, mono_langs, valid_lang_pairs)

    def build_model(self, args):
        model = super(TranslationTask, self).build_model(args)

        # xlm_add_special_tokens_to_dict_and_model(self.common_dict, model, self.mono_langs)
        xlm_add_special_tokens_to_dict_and_model(
            self.common_dict_2nd, model, self.mono_langs, extend_emb=False
        )
        assert len(self.common_dict_2nd) == len(self.common_dict) + len(self.mono_langs)

        self.sequence_generators = {}
        for mono_lang in self.mono_langs:
            self.sequence_generators[mono_lang] = SequenceGenerator(
                [model],
                tgt_dict=self.dictionary,
                beam_size=1,
                max_len_a=1.3,
                max_len_b=5,
                min_len=5,
                # keep 1 to be able to prepend bos
                max_len=model.max_decoder_positions() - 1,
            )

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.bleu_sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def _prepend_lang_bos_to_target(
        self, dataset: LanguagePairDataset, lang: str
    ) -> LanguagePairDataset:
        bos = online_backtranslation._lang_token_index(self.common_dict_2nd, lang)
        return TransformEosLangPairDataset(
            dataset,
            src_eos=self.dictionary.eos(),
            new_src_eos=self.dictionary.eos(),
            tgt_bos=self.dictionary.eos(),
            new_tgt_bos=bos,
        )

    def load_bt_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """The BT dataset is generated with (tgt, tgt) pairs.
        The actual translation to a (generated_src, tgt) pair
        is done on the fly during training.
        """
        mono_dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        assert mono_dataset is not None, f"No dataset found for {lang}"

        mono_dataset_src = PrependTokenDataset(
            mono_dataset,
            online_backtranslation._lang_token_index(self.common_dict_2nd, lang),
        )

        mono_dataset_bt = self._langpair_dataset(mono_dataset_src, mono_dataset)
        logger.info(
            f"mono_lang = {lang} "
            f"lang token index = {online_backtranslation._lang_token_index(self.common_dict_2nd, lang)} "
            f"lang token = {online_backtranslation._lang_token(lang)}"
        )

        mono_dataset_bt = self._prepend_lang_bos_to_target(mono_dataset_bt, lang)
        return mono_dataset_bt

    def load_denoise_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """Classic denoising dataset"""
        dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        noisy_dataset = NoisingDataset(
            dataset,
            self.dictionary,
            seed=1,
            max_word_shuffle_distance=self.args.max_word_shuffle_distance,
            word_dropout_prob=self.args.word_dropout_prob,
            word_blanking_prob=self.args.word_blanking_prob,
        )
        noisy_dataset = PrependTokenDataset(
            noisy_dataset,
            online_backtranslation._lang_token_index(self.common_dict_2nd, lang),
        )

        clean_dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        denoising_dataset = self._langpair_dataset(noisy_dataset, clean_dataset)
        denoising_dataset = self._prepend_lang_bos_to_target(denoising_dataset, lang)
        return denoising_dataset

    def load_translation_dataset(
        self,
        split: str,
        data_path: str,
        combine: bool = False,
        pairs: Optional[list] = None,
    ):
        # only judging with one language pair for the moment,
        # since ConcatDataset doesn't work as expected
        # assert len(self.valid_lang_pairs) == 1, "For now..."
        datasets = []
        pairs = pairs or self.valid_lang_pairs
        for i, pair in enumerate(pairs):
            # valid_lang_pair = self.valid_lang_pairs[0]
            src, tgt = pair.split("-")

            # use the same function than TranslationTask
            logger.warning(
                f"load_translation_dataset impl: {self.args.dataset_impl}, {combine=}"
            )
            src_tgt_dt = load_langpair_dataset(
                data_path,
                split,
                src,
                self.common_dict,
                tgt,
                self.common_dict,
                combine=combine,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                load_alignments=self.args.load_alignments,
                truncate_source=self.args.truncate_source,
                num_buckets=self.args.num_batch_buckets,
                shuffle=(split != "test"),
                prepend_bos_src=online_backtranslation._lang_token_index(
                    self.common_dict_2nd, src
                ),
            )

            src_tgt_eos_dt = self._prepend_lang_bos_to_target(src_tgt_dt, tgt)
            src_tgt_eos_dt.args = self.args
            datasets.append((f"{src}{tgt}", src_tgt_eos_dt))
        # return src_tgt_eos_dt
        return datasets

    def build_dataset_for_inference(
        self,
        src_tokens,
        src_lengths,
        src,
        tgt=None,
        tgt_tokens=None,
        tgt_lengths=None,
        constraints=None,
    ):
        # prepend language token to
        # NOTE src_tokens should not have any bos, eos
        src_tokens = [
            torch.cat(
                (
                    torch.LongTensor(
                        [
                            online_backtranslation._lang_token_index(
                                self.common_dict_2nd, src
                            )
                        ]
                    ),
                    x,
                    torch.LongTensor([self.dictionary.eos()]),
                )
            )
            for x in src_tokens
        ]
        src_lengths = [len(x) for x in src_tokens]

        if tgt_tokens is not None:
            tgt_tokens = [
                torch.cat((x, torch.LongTensor([self.dictionary.eos()])))
                for x in tgt_tokens
            ]
            tgt_lengths = [len(x) for x in tgt_tokens]

        src_tgt_dt = LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt=tgt_tokens,
            tgt_sizes=tgt_lengths,
            tgt_dict=self.target_dictionary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            # max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
            num_buckets=self.args.num_batch_buckets,
            constraints=constraints,
            shuffle=False,
        )

        src_tgt_eos_dt = self._prepend_lang_bos_to_target(src_tgt_dt, tgt)
        src_tgt_eos_dt.args = self.args
        return src_tgt_eos_dt

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        data = []
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))
            # REMOVE DENOISING AUTO ENCODER FOR MASS
            if len(self.lambda_dae.pieces) >= 1 and self.lambda_dae.pieces[0][1] > 0:
                data.append(
                    (f"{lang}-DENOISE", self.load_denoise_dataset(train_path, lang))
                )
            else:
                logger.warning(
                    f"Not building {lang}-/DENOISE because {self.args.lambda_dae=}"
                )

        return RoundRobinZipDatasets(OrderedDict(data))

    @property
    def symbols_to_strip(self):
        return set(
            online_backtranslation._lang_token_index(self.common_dict_2nd, x)
            for x in self.mono_langs
        )

    def display_samples_once_in_a_while(self, smp, mono_lang, other_lang):
        if 1 < self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr += 1
            return
        elif self._show_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr = 0
        else:
            self._show_samples_ctr += 1

        ln = smp["net_input"]["src_tokens"].shape[0]

        logger.info(
            f"(r:{self.args.distributed_rank}) : "
            f"{other_lang} ---> {mono_lang} "
            f"({other_lang} was generated by back-translation.) {ln} samples"
        )
        # bpe_symbol = "sentencepiece"
        # FIXME nxphi: need to change this to configureable
        bpe_symbol = "subword_nmt"
        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]

            src_str = self.common_dict_2nd.string(src_tokens, bpe_symbol)
            tgt_str = self.common_dict_2nd.string(tgt_tokens, bpe_symbol)
            logger.info(
                f"\n{i}\t\t[{other_lang} generated]  {src_str}\n"
                f"\t\t[{mono_lang} original ]  {tgt_str}\n"
                f"\t\t[ src tokens]  {src_tokens}\n"
            )

    def backtranslate_sample(self, smp, orig_lang, other_lang) -> None:
        """
        * WARNING: smp is modified in place.
        * At the start of this function, `smp` has the same input and target:
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (from data) __en__ hello world |  __en__ hello world   |
          |--------------------------------------------------------|

        * We call generator.generate(smp, bos_token = token("ro")),
        and copy the result as input
        * At the end, `smp` has the translation to other language.
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (generated) __ro__ salut lume  |  __en__ hello world   |
          |--------------------------------------------------------|

        """
        bos_token = online_backtranslation._lang_token_index(
            self.common_dict_2nd, other_lang
        )
        generated = self.sequence_generators[orig_lang].generate(
            models=[], sample=smp, bos_token=bos_token
        )

        max_lngth = max([gn[0]["tokens"].size(0) for gn in generated])
        net_input = smp["net_input"]
        n_src_tokens = torch.empty(
            size=(len(generated), max_lngth + 1), dtype=net_input["src_tokens"].dtype
        )
        n_src_lengths = torch.empty(
            len(generated), dtype=net_input["src_lengths"].dtype
        )

        for i, gn in enumerate(generated):
            tokens = gn[0]["tokens"]
            tokens_size = tokens.size(0)
            padding_needed = max_lngth - tokens_size
            tokens = torch.cat([tokens.new([bos_token]), tokens])
            tokens = F.pad(tokens, (0, padding_needed), value=self.dictionary.pad())
            n_src_tokens[i] = tokens
            n_src_lengths[i] = tokens_size + 1

        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = n_src_tokens.to(device)
        net_input["src_lengths"] = n_src_lengths.to(device)

    def get_bos_token(self, lang):
        return online_backtranslation._lang_token_index(self.common_dict_2nd, lang)

    def get_bos_token_from_sample(self, sample):
        net_input = sample["net_input"]
        source_lang_token_id = torch.unique(net_input["src_tokens"][:, 0]).item()
        source_lang_token = self.common_dict_2nd[source_lang_token_id].replace("_", "")
        target_lang_token_id = online_backtranslation._lang_token_index(
            self.common_dict_2nd, self.get_other_lang(source_lang_token)
        )

        return target_lang_token_id

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        dataset_keys = self.datasets["train"].datasets.keys()

        weights = {
            "BT": self.lambda_bt(update_num),
            "DENOISE": self.lambda_dae(update_num),
        }
        log_keys = {"BT": "bt_", "DENOISE": "dae_"}

        for dataset_key in dataset_keys:
            assert (
                dataset_key in sample or ignore_grad
            ), f"keymissing and not ignore[{ignore_grad=}]: {dataset_keys=}? {sample.keys()=}"
            if dataset_key not in sample and ignore_grad:
                # dummy batch found?
                assert all(
                    x in sample
                    for x in ["id", "nsentences", "ntokens", "net_input", "target"]
                ), f"{sample.keys()=}"
                smp = sample
                logger.warning(
                    f'invalid dummy batch found?: {sample["net_input"].keys()=}: sample: {sample}'
                )
            else:
                smp = sample[dataset_key]
            mono_lang, task_subtype = dataset_key.split("-")
            if weights[task_subtype] == 0:
                continue

            # logger.warning(f'train {dataset_key}...')
            if task_subtype == "BT":
                with torch.autograd.profiler.record_function("backtranslation"):
                    model.train(mode=self.args.bt_train)
                    # TODO: Could we translate to several language at once ?
                    # this would allow to share encoder_out and maximize GPU usage.
                    other_lang = self.get_other_lang(mono_lang)
                    self.backtranslate_sample(smp, mono_lang, other_lang)
                    self.display_samples_once_in_a_while(smp, mono_lang, other_lang)
                    model.train()
                # _src_tokens_1st = smp['net_input']['src_tokens'][:, 0]
                # _prev_output_tokens_1st = smp['net_input']['prev_output_tokens'][:, 0]
                # assert ((not torch.any(_src_tokens_1st == _prev_output_tokens_1st)) or ignore_grad
                #     ), f'Equal toks[{ignore_grad=}][rank={self.args.distributed_rank}] {_src_tokens_1st=}, {_prev_output_tokens_1st=}'

            # Like in FairseqTask.train_step
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, smp)
            loss *= weights[task_subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)

            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[task_subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(
        self,
        generator,
        models,
        sample,
        prefix_tokens=None,
        constraints=None,
        bos_token=None,
    ):
        if bos_token is None:
            # perhaps during inference
            # rely on valid-lang-pairs
            valid_lang_pair = self.valid_lang_pairs[0]
            src, tgt = valid_lang_pair.split("-")
            bos_token = online_backtranslation._lang_token_index(
                self.common_dict_2nd, tgt
            )
        assert bos_token is not None
        # logger.warning(f'inference_step: {bos_token}')
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=bos_token,
            )


@register_task("umt_online_backtranslation_xlm")
class UmtOnlineBackTranslationXLMTask(Online2ndBackTranslationXLMTask):
    """
    Same as Online2ndBackTranslationXLMTask
    Unlike OnlineBackTranslationXLMTask (version-1)
        It does not store extra lang_id into the dictionary and risk the model's predicting those tokens
        Instead, it maintains 2 dictionaries: the self.common_dict and self.common_dict_2nd
        - self.common_dict is the main one, attached to the models
        - self.common_dict_2nd is the one use to generate src_tokens and attach lang_id at the beginning of sentences
        - The XLM-based models should infer the lang-id <bos> token of the sentence and
            (1) apply the right language-embedding layer
            (2) change __langid__ ---> eos in the src_tokens
    """

    pass


@register_task("augpara_score_online2_backtranslation_xlm")
class AugParaScoreOnline2ndBackTranslationXLMTask(Online2ndBackTranslationXLMTask):
    def __init__(self, args, common_dict, mono_langs, valid_lang_pairs):
        super().__init__(args, common_dict, mono_langs, valid_lang_pairs)
        self.lambda_augpara = PiecewiseLinearFn.from_string(args.lambda_augpara)

    @staticmethod
    def add_args(parser):
        Online2ndBackTranslationXLMTask.add_args(parser)
        parser.add_argument(
            "--augpara-path", type=str, help="path to augmentation data"
        )
        parser.add_argument(
            "--augpara-pairs", type=str, help="pairs src-tgt of the augmentation data"
        )
        parser.add_argument(
            "--augpara-reverse",
            default=False,
            action="store_true",
            help="reverse each augpara data tgt->src",
        )
        parser.add_argument(
            "--lambda-augpara",
            default="1.0",
            type=str,
            metavar="N",
            help="augmentation data weight",
        )

        parser.add_argument(
            "--scores2weights",
            type=str,
            default="scale_min_max",
            help="path to augmentation data",
        )
        parser.add_argument(
            "--scores2weights-params",
            type=str,
            default="0,1",
            help="params for scores2weights",
        )
        parser.add_argument(
            "--no-use-weights",
            default=False,
            action="store_true",
            help="not using the weights",
        )

    def scores_to_weights(self, scores):
        scores2weights = self.args.scores2weights
        params = [float(x) for x in self.args.scores2weights_params.split(",")]
        logger.info(f"scores2weights params: {params}")
        if scores2weights == "scale_min_max":
            # logger.warning(f'WARNING: scale_min_max for positive similarity correlation (higher more similar), like cosine_sim, '
            #     f'for distance score, use --scores2weights neg_scale_min_max')
            # _min, _max = params[0], params[1]
            _min = params[0] if len(params) >= 1 else 0.0
            _max = params[1] if len(params) >= 2 else 1.0
            scores = np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min()) * (
                _max - _min
            ) + _min
        elif scores2weights == "neg_scale_min_max":
            # logger.warning(f'WARNING: neg_scale_min_max for negative similarity correlation (higher more similar), like distances, '
            #     f'for cosine_sim score, use --scores2weights scale_min_max')
            scores = -np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min())
        elif scores2weights == "scale_min_max_old":
            scores = np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min())
        elif scores2weights == "ones":
            weights = np.ones(shape=(len(scores,)))
        elif scores2weights == "uniform_rank":
            _min = params[0] if len(params) >= 1 else 0.0
            _max = params[1] if len(params) >= 2 else 1.0
            incr = (_max - _min) / float(len(scores))
            weights = [0] * len(scores)
            scores = np.array(scores)
            for i, idx in enumerate(np.argsort(scores)):
                weights[idx] = _min + (i + 1) * incr
            # weights = (scores - scores.min()) / (scores.max() - scores.min()) * (_max - _min) + _min
            weights = np.array(weights)
        else:
            raise ValueError(f"{scores2weights} invalid")
        return weights

    def display_samples_once_in_a_while(self, smp, mono_lang, other_lang, aug=False):
        if aug:
            if 1 < self._show_aug_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
                self._show_aug_samples_ctr += 1
                return
            elif self._show_aug_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
                self._show_aug_samples_ctr = 0
            else:
                self._show_aug_samples_ctr += 1
        else:
            if 1 < self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
                self._show_samples_ctr += 1
                return
            elif self._show_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
                self._show_samples_ctr = 0
            else:
                self._show_samples_ctr += 1

        ln = smp["net_input"]["src_tokens"].shape[0]

        if aug:
            logger.info(
                f"(r:{self.args.distributed_rank}) : "
                f"Aug translate {mono_lang} ---> {other_lang} "
                f"({other_lang} was generated by back-translation and evaluated with target.) {ln} samples"
            )
        else:
            logger.info(
                f"(r:{self.args.distributed_rank}) : "
                f"{other_lang} ---> {mono_lang} "
                f"({other_lang} was generated by back-translation.) {ln} samples"
            )
        # bpe_symbol = "sentencepiece"
        # FIXME nxphi: need to change this to configureable
        bpe_symbol = "subword_nmt"
        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]

            src_str = self.common_dict_2nd.string(src_tokens, bpe_symbol)
            tgt_str = self.common_dict_2nd.string(tgt_tokens, bpe_symbol)
            if aug:
                # hyp_tokens = smp["hyp_tokens"][i] if "hyp_tokens" in smp else None
                # hyp_str = self.common_dict_2nd.string(hyp_tokens, bpe_symbol) if hyp_tokens is not None else None
                logger.info(
                    # f"\n{i}\t\t[{other_lang} hyp]  {hyp_str}\n"
                    f"\t\t[{other_lang} tgt]  {tgt_str}\n"
                    f"\t\t[{mono_lang} src]  {src_str}\n" + (
                        f"\t\t[{mono_lang}-{other_lang} weights] {smp['weights'][i]}\n"
                        if "weights" in smp
                        else ""
                    )
                )
            else:
                logger.info(
                    f"\n{i}\t\t[{other_lang} generated]  {src_str}\n"
                    f"\t\t[{mono_lang} original ]  {tgt_str}\n"
                    f"\t\t[src tokens]  {src_tokens}\n"
                )

    def load_translation_dataset(
        self,
        split: str,
        data_path: str,
        combine: bool = False,
        lang_pair=None,
        pair_score=False,
    ):
        # only judging with one language pair for the moment,
        # since ConcatDataset doesn't work as expected
        def build_dataset(_src, _tgt):
            # use the same function than TranslationTask
            if pair_score and self.args.no_use_weights:
                logger.info(f"Pair-Score: NO USE AUG_DATA WEIGHTS")
            if pair_score and not self.args.no_use_weights:
                pair_score_path = os.path.join(data_path, "index.pth")
                assert os.path.exists(pair_score_path), f"{pair_score_path} not found."
                logger.info(
                    f"Load aug data index {pair_score_path}, {self.args.scores2weights=}"
                )
                pth = torch.load(pair_score_path)
                scores = pth["scores"]
                weights = self.scores_to_weights(scores)

                src_tgt_dt = load_langpair_weights_dataset(
                    data_path=data_path,
                    split=split,
                    weights=weights,
                    src=_src,
                    src_dict=self.common_dict,
                    tgt=_tgt,
                    tgt_dict=self.common_dict,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    load_alignments=self.args.load_alignments,
                    truncate_source=self.args.truncate_source,
                    num_buckets=self.args.num_batch_buckets,
                    shuffle=(split != "test"),
                    prepend_bos_src=online_backtranslation._lang_token_index(
                        self.common_dict_2nd, _src
                    ),
                )
            else:
                src_tgt_dt = load_langpair_dataset(
                    data_path=data_path,
                    split=split,
                    src=_src,
                    src_dict=self.common_dict,
                    tgt=_tgt,
                    tgt_dict=self.common_dict,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    load_alignments=self.args.load_alignments,
                    truncate_source=self.args.truncate_source,
                    num_buckets=self.args.num_batch_buckets,
                    shuffle=(split != "test"),
                    prepend_bos_src=online_backtranslation._lang_token_index(
                        self.common_dict_2nd, _src
                    ),
                )

            src_tgt_eos_dt = self._prepend_lang_bos_to_target(src_tgt_dt, _tgt)
            src_tgt_eos_dt.args = self.args
            return src_tgt_eos_dt

        if split == "train":
            assert lang_pair is not None
            src, tgt = lang_pair.split("-")
            return build_dataset(src, tgt)
        else:
            assert split in ["valid", "test"]
            datasets = []
            for i, pair in enumerate(self.valid_lang_pairs):
                src, tgt = pair.split("-")
                dataset = build_dataset(src, tgt)
                datasets.append((f"{src}{tgt}", dataset))
            return datasets

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        data = []
        args = self.args
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))
            # REMOVE DENOISING AUTO ENCODER FOR MASS
            if len(self.lambda_dae.pieces) >= 1 and self.lambda_dae.pieces[0][1] > 0:
                data.append(
                    (f"{lang}-DENOISE", self.load_denoise_dataset(train_path, lang))
                )
            else:
                logger.info(
                    f"Not building {lang}-/DENOISE because {self.args.lambda_dae=}"
                )
            # aug data
        augpara_paths = args.augpara_path.split(",")
        augpara_pairs = args.augpara_pairs.split(",")
        assert len(augpara_paths) == len(
            augpara_pairs
        ), f"{len(augpara_paths)=} != {len(augpara_pairs)}"
        for i, (p_path, p_pair) in enumerate(zip(augpara_paths, augpara_pairs)):
            # aug_path = os.path.join(p_path, f'train.{p_pair}')
            aug_path = p_path
            src, tgt = p_pair.split("-")
            logger.info(f"Loading aug data: {p_pair} at {aug_path}")
            dataset = self.load_translation_dataset(
                "train", aug_path, lang_pair=p_pair, pair_score=True
            )
            data.append((f"{src}{tgt}-AUG", dataset))
            if args.augpara_reverse:
                logger.info(f"Reversing aug data: {p_pair} at {aug_path}")
                r_dataset = self.load_translation_dataset(
                    "train", aug_path, lang_pair=f"{tgt}-{src}", pair_score=True
                )
                data.append((f"{tgt}{src}-AUG", r_dataset))

        return RoundRobinZipDatasets(OrderedDict(data))

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        dataset_keys = self.datasets["train"].datasets.keys()

        weights = {
            "BT": self.lambda_bt(update_num),
            "DENOISE": self.lambda_dae(update_num),
            "AUG": self.lambda_augpara(update_num),
        }
        log_keys = {"BT": "bt_", "DENOISE": "dae_", "AUG": "aug_"}

        for dataset_key in dataset_keys:
            smp = sample[dataset_key]
            mono_lang, task_subtype = dataset_key.split("-")
            if weights[task_subtype] == 0:
                continue

            if task_subtype == "BT":
                with torch.autograd.profiler.record_function("backtranslation"):
                    model.train(mode=self.args.bt_train)
                    # TODO: Could we translate to several language at once ?
                    # this would allow to share encoder_out and maximize GPU usage.
                    other_lang = self.get_other_lang(mono_lang)
                    self.backtranslate_sample(smp, mono_lang, other_lang)
                    self.display_samples_once_in_a_while(smp, mono_lang, other_lang)
                    model.train()

            # Like in FairseqTask.train_step
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, smp)
            loss *= weights[task_subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)

            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[task_subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        aug_sample_size = sum(x.get("aug_sample_size", 0) for x in logging_outputs)
        if aug_sample_size:
            aug_loss_sum = sum(x.get("aug_loss", 0) for x in logging_outputs)
            aug_loss_sum *= 1 / aug_sample_size / math.log(2)
            metrics.log_scalar("aug_loss", aug_loss_sum, aug_sample_size, round=3)

            aug_nll_loss_sum = sum(x.get("aug_nll_loss", 0) for x in logging_outputs)
            aug_ntokens = sum(x.get("aug_ntokens", 0) for x in logging_outputs)
            aug_nll_loss_sum *= 1 / aug_ntokens / math.log(2)
            metrics.log_scalar("aug_nll_loss", aug_nll_loss_sum, aug_ntokens, round=3)
            metrics.log_derived(
                "aug_ppl",
                lambda meters: utils.get_perplexity(meters["aug_nll_loss"].avg),
            )


@register_task("umt_augpara_score_online_backtranslation_xlm")
class UmtAugParaScoreOnlineBackTranslationXLMTask(
    AugParaScoreOnline2ndBackTranslationXLMTask
):
    """
    Same as AugParaScoreOnline2ndBackTranslationXLMTask
    Unlike OnlineBackTranslationXLMTask (version-1)
        It does not store extra lang_id into the dictionary and risk the model's predicting those tokens
        Instead, it maintains 2 dictionaries: the self.common_dict and self.common_dict_2nd
        - self.common_dict is the main one, attached to the models
        - self.common_dict_2nd is the one use to generate src_tokens and attach lang_id at the beginning of sentences
        - The XLM-based models should infer the lang-id <bos> token of the sentence and
            (1) apply the right language-embedding layer
            (2) change __langid__ ---> eos in the src_tokens
    """

    pass
