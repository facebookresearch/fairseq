# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
# from fairseq.tasks.xlm_code_tasks import load_langpair_weights_dataset
import itertools
from fairseq.data.strip_token_dataset import StripTokenDataset
from fairseq.data.shorten_dataset import TruncateDataset
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.append_token_dataset import AppendTokenDataset
import json
import logging
import math
import os
from argparse import Namespace
from collections import OrderedDict, defaultdict
from pathlib import Path
from argparse import ArgumentError
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fairseq
from fairseq import metrics, options, utils
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    NoisingDataset,
    PrependTokenDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
    data_utils,
    encoders,
    indexed_dataset,
    language_pair_dataset,
)
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks import register_task
from fairseq.tasks.translation import EVAL_BLEU_ORDER, TranslationTask, load_langpair_dataset

from ..mbart_sequence_generator import OBTmBARTSequenceGenerator
from ..data.swav_dataset import load_langpair_weights_dataset

from fairseq.tasks import online_backtranslation 
from fairseq.tasks.online_backtranslation import PiecewiseLinearFn, OnlineBackTranslationTask
from fairseq.tasks.online_backtranslation import add_special_tokens_to_dict_and_model

logger = logging.getLogger(__name__)


def _mbart_lang_token(lang: str) -> str:
    return f"[{lang}]"


def _mbart_lang_token_index(dictionary, lang: str) -> int:
    return dictionary.index(_mbart_lang_token(lang))


def mbart_load_langpair_dataset(
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
    # append_bos_src=None,
    # prepend_bos_tgt=None,
    # append_bos_tgt=None,
    append_lang_src=None,
    append_lang_tgt=None,
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

    assert append_lang_src is not None and append_lang_tgt is not None
    if append_lang_src:
        src_dataset = AppendTokenDataset(
            src_dataset, append_lang_src
        )
    if append_lang_tgt:
        tgt_dataset = AppendTokenDataset(
            tgt_dataset, append_lang_tgt
        )
        eos = append_lang_tgt

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return MBartLanguagePairDatset(
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
    )


class MBartLanguagePairDatset(LanguagePairDataset):
    def collater(self, samples, pad_to_length=None):
        res = language_pair_dataset.collate(
            samples,
            pad_idx=self.src_dict.pad(),
            # eos_idx=self.eos,
            eos_idx=None,   # NOTE eos=None so move_eos_to_beginning will move <lang-tok> to beginning
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens.device)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens.device)
                )
                res["net_input"]["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens.device)
                )
        return res


@register_task("online_backtranslation_from_pretrained_bart")
class OnlineBackTranslationFromPretBARTTask(OnlineBackTranslationTask):
    """
    OnlineBackTranslationMBARTTask different from OnlineBackTranslationTask:
    -- OnlineBackTranslationTask
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
    ***** OnlineBackTranslationFromPretBARTTask taken from TranslationFromPretrainedBARTTask
    from https://github.com/pytorch/fairseq/tree/master/examples/mbart
        --prepend-bos not activated
            --> we should try both
    -- OnlineBackTranslationMBARTTask
        * Should follow translation_from_pretrained_bart
        * NOTE-1: --add-lang-token --langs and added into the vocabulary
            * NOTE-1.1: expect no change in vocabulary after add_secial_tokens_to_dict_and_model
        * NOTE-2: __langtok__ is added at start of enc and start/end of dec in/out
            * need to rebuild dataset
        * WARNING: smp is modified in place.
        * At the start of this function, `smp` has the same input and target:
          |-----------------------------------------------------------------|
          | smp['net_input']['src_tokens']       | smp['target']            |
          | (from data) hello world </s> [en_XX] | [en_XX] hello world </s> |
          |-----------------------------------------------------------------|

        * We call generator.generate(smp, bos_token = token("ro")),
        and copy the result as input
        * At the end, `smp` has the translation to other language.
          |-----------------------------------------------------------------|
          | smp['net_input']['src_tokens']       | smp['target']            |
          | (generated) salut lume </s> [ro_RO]  | [en_XX] hello world </s> |
          |-----------------------------------------------------------------|
    """
    def __init__(self, args, common_dict, mono_langs, langs, valid_lang_pairs, freq_dicts=None):
        super().__init__(args, common_dict, mono_langs, valid_lang_pairs)
        self.freq_dicts = freq_dicts
        self.langs = langs
        self.indic_tokenizer = None
        self.indic_langs = None
        self.init_indic_tokenizer()

    def init_indic_tokenizer(self):
        indic_path = getattr(self.args, "indic_nlp_path", "")
        indic_langs = getattr(self.args, "indic_langs", "").split(",")
        if indic_path is not None and os.path.exists(indic_path) and len(indic_langs) > 0:
            # FIXME: this can only work on python2
            self.indic_path = indic_path
            self.indic_langs = indic_langs
            self.indic_tokenizer = {}
            logger.info(f'Loading Indic NLP path for {indic_langs}: {indic_path}')
            try:
                import sys
                sys.path.extend([
                    indic_path,
                    os.path.join(indic_path, "src"),
                ])
                from indicnlp.tokenize import indic_tokenize
                from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            except Exception:
                raise Exception(
                    "Cannot load Indic NLP Library, make sure --indic-nlp-path is correct, got {}".format(
                        indic_path
                    )
                )
            # create normalizer
            for lang in indic_langs:
                sim_lang = lang[:2]
                factory = IndicNormalizerFactory()
                normalizer = factory.get_normalizer(
                    sim_lang, remove_nuktas=getattr(self.args, "indic_remove_nuktas", False)
                )

                def tok_fn(line):
                    line = normalizer.normalize(line)
                    line = " ".join(indic_tokenize.trivial_tokenize(line, sim_lang))
                    oline = line
                    return oline

                self.indic_tokenizer[lang] = tok_fn
                logger.info(f'Loading Indic tokenizer for {lang} -> {sim_lang}')

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        OnlineBackTranslationTask.add_args(parser)
        # from multilignual denoising
        parser.add_argument("--add-lang-token", default=False, action="store_true")
        parser.add_argument('--langs', type=str, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        parser.add_argument('--bwd-eval-tokenized-bleu', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        parser.add_argument('--indic-nlp-path', type=str, default=None,
                            help='path to indic tokenizer, for sinhala and nepali languages; NOTE, value of BLEU with this is not correct')
        parser.add_argument('--indic-langs', type=str, default="ne_NP",
                            help='name of indic langs, which will use indic path to tokenize and compute bleu')
        parser.add_argument('--indic-remove-nuktas', action='store_true',
                            help='remove-nuktas')
        # disable target tokens according to top-frequency of weights
        parser.add_argument('--top-frequency', default=1, type=float, metavar='N',
                            help='During BT, only generate tokens appearing > top-freq in target corpus.'
                                 'Data path must contains dict dict.freq.<lang>.txt specifying the frequencies of all tokens'
                                 ' for that language corpus'
                            ),
        parser.add_argument('--no-top-freq-after', default=1000, type=int, metavar='N',
                            help='Stop top-frequency inferencing after number of updates'),

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        assert args.mono_langs is not None

        mono_langs = args.mono_langs.split(",")
        valid_lang_pairs = args.valid_lang_pairs.split(",")

        # load dictionary
        dict_path = os.path.join(paths[0], "dict.txt")
        common_dict = cls.load_dictionary(dict_path)

        # NOTE multilingual data path (roberta/mbart) / multilignual denoising
        data_path = paths[0]
        if args.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
            raise NotImplementedError(f'--langs should have list of langs, instead of None')
        else:
            languages = args.langs.split(",")
        logger.info(f'Automatically add_lang_token: {languages=} and mask')
        for lang in languages:
            common_dict.add_symbol(_mbart_lang_token(lang))
        common_dict.add_symbol("<mask>")

        # frequency-based dictionary
        freq_dicts = {}
        for mono_lang in mono_langs:
            freq_dict_path = os.path.join(paths[0], f"dict.freq.{mono_lang}.txt")
            if os.path.exists(freq_dict_path):
                try:
                    freq_dict = cls.load_dictionary(freq_dict_path)
                    for lang in languages:
                        freq_dict.add_symbol(_mbart_lang_token(lang))
                    freq_dict.add_symbol("<mask>")
                    freq_dicts[mono_lang] = freq_dict
                    logger.info(f'Load Top freq dictionary dict {mono_lang}')
                except Exception as e:
                    logger.warning(f'Error Frequency dict {mono_lang}: {e}')
                    freq_dicts = {}
                    break
            else:
                logger.warning(f'Frequency dict {mono_lang} not found {freq_dict_path}, not using this --top-frequency')
                freq_dicts = {}
                break

        return cls(args, common_dict, mono_langs, languages, valid_lang_pairs, freq_dicts)

    def dict_string(
        self,
        dictionary,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" "
    ):
        """
        Customized dict-string, include all special tokens for debugging purpose
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.dict_string(dictionary, t, bpe_symbol, escape_unk, extra_symbols_to_ignore, include_eos=include_eos)
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])

        def token_string(i):
            if i == dictionary.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return dictionary.unk_string(escape_unk)
            else:
                return dictionary[i]

        sent = separator.join(
            token_string(i)
            for i in tensor
            if utils.item(i) not in extra_symbols_to_ignore
        )

        return data_utils.post_process(sent, bpe_symbol)

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
        bpe_symbol = "sentencepiece"
        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]

            src_str = self.dict_string(self.dictionary, src_tokens, bpe_symbol)
            tgt_str = self.dict_string(self.dictionary, tgt_tokens, bpe_symbol)
            logger.info(
                f"\n{i}\t\t[{other_lang} generated]  {src_str}\n"
                f"\t\t[{mono_lang} original ]  {tgt_str}\n"
                f"\t\t[ src tokens]  {src_tokens}\n"
            )

    def _langpair_dataset(
        self, src: FairseqDataset, tgt: FairseqDataset, eos: int = None
    ) -> LanguagePairDataset:
        # enable append_source_id
        # enable prepend_bos according to args.prepend_bos
        return LanguagePairDataset(
            src,
            src.sizes,
            self.dictionary,
            tgt=tgt,
            tgt_sizes=tgt.sizes,
            tgt_dict=self.dictionary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            # TODO: should we shuffle ? we are already sorting batch by sizes so ?
            # shuffle=True,
            eos=eos,
        )

    def load_bt_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """The BT dataset is generated with (tgt, tgt) pairs.
        The actual translation to a (generated_src, tgt) pair
        is done on the fly during training.
        src' = [src, lang_token]
        tgt' = [lang_token, src]
        """
        mono_dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        assert mono_dataset is not None, f"No dataset found for {lang}"

        # enable append_source_id
        # enable prepend_bos according to args.prepend_bos
        mono_dataset_src = AppendTokenDataset(
            mono_dataset, _mbart_lang_token_index(self.dictionary, lang)
        )

        mono_dataset_bt = self._langpair_dataset(
            mono_dataset_src, mono_dataset_src, eos=_mbart_lang_token_index(self.dictionary, lang))
        # input  src: [src, eos, langtok], tgt: [tgt, eos, langtok]
        # output src: [src, eos, langtok], tgt: <prev_tokens: [langtok, tgt, eos], targets: [tgt,eos,langtok]>
        logger.info(
            f"mono_lang = {lang} "
            f"lang token index = {_mbart_lang_token_index(self.dictionary, lang)} "
            f"lang token = {_mbart_lang_token(lang)}"
        )
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
        noisy_dataset = AppendTokenDataset(
            noisy_dataset, _mbart_lang_token_index(self.dictionary, lang)
        )

        clean_dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        clean_dataset = AppendTokenDataset(
            clean_dataset, _mbart_lang_token_index(self.dictionary, lang)
        )
        denoising_dataset = self._langpair_dataset(
            noisy_dataset, clean_dataset, eos=_mbart_lang_token_index(self.dictionary, lang))
        return denoising_dataset

    def sample_switch_srctgt(self, sample):
        # switch the src and tgt of sample
        # for valid translation validation, used to translate src->tgt,tgt->src at the same time
        """
        batch = {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths, "prev_output_tokens": prev_output_tokens},
            "target": target,
        }
        src_tokens : a b c </s> [en_XX]
        prev_output: [ne_NP] d e f </s>
        target:    : d e f </s> [ne_NP]
        ->
        src_tokens : d e f </s> [ne_NP]
        prev_output: [en_XX] a b c
        target:    : a b c </> [en_XX]
        """
        pad = self.dictionary.pad()
        # eos = self.dictionary.eos()
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        target = sample['target']
        target_lengths = (target != pad).long().sum(-1)
        assert torch.all(src_lengths == (src_tokens != pad).long().sum(-1)), f'src-lengths diff {src_lengths}, {src_tokens}'

        o_src_tokens = target.clone()
        o_src_lengths = target_lengths.clone()
        o_target = src_tokens.clone()

        o_prev_output_tokens = o_target.clone()
        o_prev_lengths = src_lengths.clone()
        for i in range(o_prev_output_tokens.size(0)):
            sent = o_target[i, :o_prev_lengths[i]]
            o_prev_output_tokens[i, :o_prev_lengths[i]] = torch.cat((sent[-1:], sent[:-1]), 0)

        sample['net_input']['src_tokens'] = o_src_tokens
        sample['net_input']['src_lengths'] = o_src_lengths
        sample['net_input']['prev_output_tokens'] = o_prev_output_tokens
        sample['target'] = o_target

        return sample

    def load_translation_dataset(
        self, split: str, data_path: str, combine: bool = False, lang_pair=None,
    ):
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
                # adapted from translation_from_pretrained_bart
                prepend_bos=getattr(self.args, "prepend_bos", False),
                append_source_id=True,
            )
            src_tgt_eos_dt = src_tgt_dt
            src_tgt_eos_dt.args = self.args
            return src_tgt_eos_dt

        if split == 'train':
            assert lang_pair is not None
            src, tgt = lang_pair.split('-')
            return build_dataset(src, tgt)
        else:
            assert split in ['valid', 'test']
            datasets = []
            for i, pair in enumerate(self.valid_lang_pairs):
                src, tgt = pair.split("-")
                dataset = build_dataset(src, tgt)
                datasets.append((f'{src}{tgt}', dataset))
            return datasets

    def build_generator(self, models, args, eos=None, lang=None, **unused):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            )
        else:
            return OBTmBARTSequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                # eos=eos if eos is not None else self.tgt_dict.index("[{}]".format(self.args.target_lang)),
                eos=eos,
                top_frequency=getattr(args, "top_frequency", 1),
                top_freq_dict=self.freq_dicts[lang] if self.freq_dicts is not None and lang in self.freq_dicts else None,
            )

    def build_bt_generator(self, model, other_lang, **kwargs):
        generator = SequenceGenerator(
            [model],
            tgt_dict=self.dictionary,
            beam_size=1,
            max_len_a=1.3,
            max_len_b=5,
            min_len=5,
            # keep 1 to be able to prepend bos
            max_len=model.max_decoder_positions() - 1,
            eos=self.dictionary.index("[{}]".format(other_lang)),
        )
        return generator

    def build_model(self, args):
        # torch.autograd.set_detect_anomaly(True)
        model = super(TranslationTask, self).build_model(args)

        add_special_tokens_to_dict_and_model(
            self.common_dict, model, self.langs,
            lang_tok_fn=_mbart_lang_token,
            mask_tok_before=False,
        )

        self.sequence_generators = {}
        for mono_lang in self.mono_langs:
            other_lang = self.get_other_lang(mono_lang)
            self.sequence_generators[mono_lang] = self.build_bt_generator(model, other_lang)

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            eval_bleu_detok = getattr(args, "eval_bleu_detok", None)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=eval_bleu_detok, **detok_args
                )
            )
            logger.info(
                f'Eval configs: {eval_bleu_detok=}: {self.tokenizer}; '
                f'{getattr(self.cfg, "eval_tokenized_bleu", False)=}'
                f'{getattr(self.args, "bwd_eval_tokenized_bleu", False)=}'
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.bleu_sequence_generator = {}
            for mono_lang in self.mono_langs:
                other_lang = self.get_other_lang(mono_lang)
                self.bleu_sequence_generator[mono_lang] = self.build_generator(
                    [model], Namespace(**gen_args),
                    eos=self.dictionary.index("[{}]".format(other_lang)),
                    lang=other_lang
                )
        return model

    def backtranslate_sample(self, smp, orig_lang, other_lang, update_num) -> None:
        """
        * WARNING: smp is modified in place.
        * At the start of this function, `smp` has the same input and target:
          |---------------------------------------------------------------------------------------------|
          | smp['net_input']['src_tokens']    | smp['net_input']['prev_output_tokens'] | smp['target]   |
          | (from data) hello world </s> [en] | [en] hello world </s>                  | hello world </s> [en]
          |---------------------------------------------------------------------------------------------|

        * We call generator.generate(smp, bos_token = token("ro")),
        and copy the result as input
        * At the end, `smp` has the translation to other language.
          |---------------------------------------------------------------------------------------------|
          | smp['net_input']['src_tokens']    | smp['net_input']['prev_output_tokens'] | smp['target]   |
          | (from data) salut lume </s> [ro]  | [en] hello world </s>                  | hello world </s> [en]
          |---------------------------------------------------------------------------------------------|

        """
        bos_token = _mbart_lang_token_index(self.dictionary, other_lang)
        generated = self.sequence_generators[orig_lang].generate(
            models=[], sample=smp, bos_token=bos_token,
            no_freq_mask=update_num > self.args.no_top_freq_after or self.args.top_frequency < 1.0
        )

        max_length = max([gn[0]["tokens"].size(0) for gn in generated])
        net_input = smp["net_input"]
        n_src_tokens = torch.empty(
            size=(len(generated), max_length), dtype=net_input["src_tokens"].dtype
        )
        n_src_lengths = torch.empty(
            len(generated), dtype=net_input["src_lengths"].dtype
        )

        for i, gn in enumerate(generated):
            tokens = gn[0]["tokens"]
            tokens_size = tokens.size(0)
            padding_needed = max_length - tokens_size
            tokens = F.pad(tokens, (0, padding_needed), value=self.dictionary.pad())
            n_src_tokens[i] = tokens
            n_src_lengths[i] = tokens_size

        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = n_src_tokens.to(device)
        net_input["src_lengths"] = n_src_lengths.to(device)

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        data = []
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))
            if len(self.lambda_dae.pieces) >= 1 and self.lambda_dae.pieces[0][1] > 0:
                data.append(
                    (f"{lang}-DENOISE", self.load_denoise_dataset(train_path, lang))
                )
            else:
                logger.info(f'Not building {lang}-/DENOISE because {self.args.lambda_dae=}')

        return RoundRobinZipDatasets(OrderedDict(data))

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, bos_token=None
    ):
        if bos_token is None:
            # perhaps during inference as bos_token None
            # rely on valid-lang-pairs to infer bos_token
            valid_lang_pair = self.valid_lang_pairs[0]
            src, tgt = valid_lang_pair.split("-")
            bos_token = _mbart_lang_token_index(self.dictionary, tgt)
        assert bos_token is not None
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints, bos_token=bos_token,
            )

    def _inference_with_bleu(self, generator, sample, model, eval_tokenized_bleu=None):
        import sacrebleu
        langs = self.args.langs.split(",")
        ignore_lang_indices = [self.dictionary.index(f'[{x}]') for x in langs]
        extra_symbols_to_ignore = [generator.eos] + ignore_lang_indices

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
                extra_symbols_to_ignore=extra_symbols_to_ignore,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        # extract bos_token:
        bos_token = None
        if "net_input" in sample and "prev_output_tokens" in sample['net_input']:
            prev_output_tokens = sample['net_input']['prev_output_tokens']
            assert (
                torch.all(prev_output_tokens[:, 0] == prev_output_tokens[0, 0])
            ), f'inconsistent prev_tokens: {prev_output_tokens[:, 0]}'
            bos_token = prev_output_tokens[0, 0].item()
        assert bos_token is not None
        bos_slang = self.tgt_dict[bos_token].replace("[", "").replace("]", "")

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None, bos_token=bos_token)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )

        if self.indic_tokenizer is not None and bos_slang in self.indic_tokenizer:
            tokenizer = self.indic_tokenizer[bos_slang]
            hyps = [tokenizer(x) for x in hyps]
            refs = [tokenizer(x) for x in refs]

        if self.cfg.eval_bleu_print_samples:
            lang = self.tgt_dict[generator.eos]
            logger.info(f"example hypothesis {lang}: " + hyps[0])
            logger.info(f"example reference  {lang}: " + refs[0])

        eval_tokenized_bleu = self.cfg.eval_tokenized_bleu if eval_tokenized_bleu is None else eval_tokenized_bleu
        if eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def valid_step(self, sample, model, criterion):
        # super(TranslationTask, self) to bypass TranslationTask valid_step
        loss, sample_size, logging_output = super(TranslationTask, self).valid_step(sample, model, criterion)
        if getattr(self.args, "eval_bleu", False):
            # forward mt
            mono_langs = self.mono_langs
            bleu = self._inference_with_bleu(
                self.bleu_sequence_generator[mono_langs[0]], sample, model,
            )
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

            # backward blue
            if self.valid_mt_rev:
                bwd_sample = self.sample_switch_srctgt(sample)
                bwd_bleu = self._inference_with_bleu(
                    self.bleu_sequence_generator[mono_langs[1]], bwd_sample, model,
                    eval_tokenized_bleu=getattr(self.args, "bwd_eval_tokenized_bleu", False),
                )
                logging_output["_bwd_bleu_sys_len"] = bwd_bleu.sys_len
                logging_output["_bwd_bleu_ref_len"] = bwd_bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bwd_bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bwd_bleu_counts_" + str(i)] = bwd_bleu.counts[i]
                    logging_output["_bwd_bleu_totals_" + str(i)] = bwd_bleu.totals[i]
        return loss, sample_size, logging_output

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
            smp = sample[dataset_key]
            mono_lang, task_subtype = dataset_key.split("-")
            if weights[task_subtype] == 0:
                continue

            if task_subtype == "BT":
                with torch.autograd.profiler.record_function("backtranslation"):
                    model.eval()
                    # TODO: Could we translate to several language at once ?
                    # this would allow to share encoder_out and maximize GPU usage.
                    other_lang = self.get_other_lang(mono_lang)
                    self.backtranslate_sample(smp, mono_lang, other_lang, update_num)
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


@register_task("augpara_online_backtranslation_from_pretrained_bart")
class AugParaOnlineBackTranslationFromPretBARTTask(OnlineBackTranslationFromPretBARTTask):
    """
    Augmentation with Mined Pseudo-parallel dataset with OnlineBackTranslationFromPretBARTTask
    """
    def __init__(self, args, common_dict, mono_langs, langs, valid_lang_pairs, freq_dicts=None):
        super().__init__(args, common_dict, mono_langs, langs, valid_lang_pairs, freq_dicts)
        self.lambda_augpara = PiecewiseLinearFn.from_string(args.lambda_augpara)

    @staticmethod
    def add_args(parser):
        OnlineBackTranslationFromPretBARTTask.add_args(parser)
        parser.add_argument('--augpara-path', type=str,
                            help='path to augmentation data')
        parser.add_argument('--augpara-pairs', type=str,
                            help='pairs src-tgt of the augmentation data')
        parser.add_argument('--augpara-reverse', default=False, action='store_true',
                            help='reverse each augpara data tgt->src')
        parser.add_argument('--lambda-augpara', default="1.0", type=str, metavar='N',
                            help='augmentation data weight')

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
                logger.info(f'Not building {lang}-/DENOISE because {self.args.lambda_dae=}')
            # aug data
        augpara_paths = args.augpara_path.split(',')
        augpara_pairs = args.augpara_pairs.split(',')
        assert len(augpara_paths) == len(augpara_pairs), f'{len(augpara_paths)=} != {len(augpara_pairs)}'
        for i, (p_path, p_pair) in enumerate(zip(augpara_paths, augpara_pairs)):
            aug_path = p_path
            src, tgt = p_pair.split('-')
            logger.info(f'Loading aug data: {p_pair} at {aug_path}')
            dataset = self.load_translation_dataset('train', aug_path, lang_pair=p_pair)
            data.append((f'{src}{tgt}-AUG', dataset))
            if args.augpara_reverse:
                logger.info(f'Reversing aug data: {p_pair} at {aug_path}')
                r_dataset = self.load_translation_dataset('train', aug_path, lang_pair=f'{tgt}-{src}')
                data.append((f'{tgt}{src}-AUG', r_dataset))

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
                    model.eval()
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
        if aug_sample_size > 0:
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


@register_task("top_freq_online_backtranslation_from_pretrained_bart")
class TopFreqOnlineBackTranslationFromPretBARTTask(OnlineBackTranslationFromPretBARTTask):
    """
    Top word frequency Online backtranslation
    - During BT, only predicts tokens that appear >99% of the target corpus
    - Similar to how mBART was trained
    Already Merged with OnlineBackTranslationFromPretBARTTask
    """
    pass


@register_task("top_freq_dy_v3_augpara_score_online_backtranslation_from_pretrained_bart")
class TopFreqDynamicV3AugParaOnlineBackTranslationFromPretBARTTask(OnlineBackTranslationFromPretBARTTask):
    """
    Optinally Strict version in the back-translation process
        enforce that in the predicted back-translations, the </s> are present at the end of target
    Merged with top_freq_dy v1 and v2 AugParaOnlineBackTranslationFromPretBARTTask
    """
    def __init__(self, args, common_dict, mono_langs, langs, valid_lang_pairs, freq_dicts=None):
        super().__init__(args, common_dict, mono_langs, langs, valid_lang_pairs, freq_dicts)
        self.lambda_augpara = PiecewiseLinearFn.from_string(args.lambda_augpara)

    @staticmethod
    def add_args(parser):
        OnlineBackTranslationFromPretBARTTask.add_args(parser)
        parser.add_argument('--augpara-path', type=str,
                            help='path to augmentation data')
        parser.add_argument('--augpara-pairs', type=str,
                            help='pairs src-tgt of the augmentation data')
        parser.add_argument('--augpara-reverse', default=False, action='store_true',
                            help='reverse each augpara data tgt->src')
        parser.add_argument('--lambda-augpara', default="1.0", type=str, metavar='N',
                            help='augmentation data weight')
        parser.add_argument('--scores2weights', type=str, default='scale_min_max',
                            help='type of scoring alignment scores of mined data to loss weights')
        parser.add_argument('--scores2weights-params', type=str, default='0,1',
                            help='params for scores2weights')
        parser.add_argument('--no-use-weights', default=False, action='store_true',
                            help='not using the weights at all, disabling scores2weights ')
        parser.add_argument('--no-bt-strict', default=False, action='store_true',
                            help='not enable strict mode for BT process ')

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        aug_sample_size = sum(x.get("aug_sample_size", 0) for x in logging_outputs)
        if aug_sample_size > 0:
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
                    model.eval()
                    # model.train(mode=self.args.bt_train) FIXME: this one not activated here yet
                    # TODO: Could we translate to several language at once ?
                    # this would allow to share encoder_out and maximize GPU usage.
                    other_lang = self.get_other_lang(mono_lang)
                    self.backtranslate_sample(smp, mono_lang, other_lang, update_num)
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

    def scores_to_weights(self, scores):
        scores2weights = self.args.scores2weights
        params = [float(x) for x in self.args.scores2weights_params.split(",")]
        logger.info(f'scores2weights params: {params}')
        if scores2weights == 'scale_min_max':
            logger.warning(
                f'WARNING: scale_min_max for positive similarity correlation (higher more similar), like cosine_sim, '
                f'for distance score, use --scores2weights neg_scale_min_max')
            _min = params[0] if len(params) >= 1 else 0.0
            _max = params[1] if len(params) >= 2 else 1.0
            scores = np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min()) * (_max - _min) + _min
        elif scores2weights == 'neg_scale_min_max':
            logger.warning(
                f'WARNING: neg_scale_min_max for negative similarity correlation (higher more similar), like distances, '
                f'for cosine_sim score, use --scores2weights scale_min_max')
            scores = -np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min())
        elif scores2weights == 'scale_min_max_old':
            scores = np.array(scores)
            weights = (scores - scores.min()) / (scores.max() - scores.min())
        elif scores2weights == 'ones':
            weights = np.ones(shape=(len(scores,)))
        elif scores2weights == "uniform_rank":
            _min = params[0] if len(params) >= 1 else 0.0
            _max = params[1] if len(params) >= 2 else 1.0
            incr = (_max - _min) / float(len(scores))
            weights = [0] * len(scores)
            scores = np.array(scores)
            for i, idx in enumerate(np.argsort(scores)):
                weights[idx] = _min + (i + 1) * incr
            weights = np.array(weights)
        else:
            raise ValueError(f'{scores2weights} invalid')
        return weights

    def load_translation_dataset(
        self, split: str, data_path: str, combine: bool = False, lang_pair=None, pair_score=False,
    ):

        def build_dataset(_src, _tgt):
            # use the same function than TranslationTask
            if pair_score and self.args.no_use_weights:
                logger.info(f'Pair-Score: NO USE AUG_DATA WEIGHTS')
            if pair_score and not self.args.no_use_weights:
                pair_score_path = os.path.join(data_path, "index.pth")
                assert os.path.exists(pair_score_path), f'{pair_score_path} not found.'
                logger.info(f'Load aug data index {pair_score_path}, {self.args.scores2weights=}')
                pth = torch.load(pair_score_path)
                scores = pth['scores']
                weights = self.scores_to_weights(scores)
                src_tgt_dt = load_langpair_weights_dataset(
                    data_path,
                    split,
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
                    prepend_bos=getattr(self.args, "prepend_bos", False),
                    append_source_id=True,
                )
            else:
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
                    prepend_bos=getattr(self.args, "prepend_bos", False),
                    append_source_id=True,
                )
            src_tgt_eos_dt = src_tgt_dt
            src_tgt_eos_dt.args = self.args
            return src_tgt_eos_dt

        if split == 'train':
            assert lang_pair is not None
            src, tgt = lang_pair.split('-')
            return build_dataset(src, tgt)
        else:
            assert split in ['valid', 'test']
            datasets = []
            for i, pair in enumerate(self.valid_lang_pairs):
                src, tgt = pair.split("-")
                dataset = build_dataset(src, tgt)
                datasets.append((f'{src}{tgt}', dataset))
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
                logger.info(f'Not building {lang}-/DENOISE because {self.args.lambda_dae=}')
            # aug data
        augpara_paths = args.augpara_path.split(',')
        augpara_pairs = args.augpara_pairs.split(',')
        assert len(augpara_paths) == len(augpara_pairs), f'{len(augpara_paths)=} != {len(augpara_pairs)}'
        for i, (p_path, p_pair) in enumerate(zip(augpara_paths, augpara_pairs)):
            aug_path = p_path
            src, tgt = p_pair.split('-')
            logger.info(f'Loading aug data: {p_pair} at {aug_path}')
            dataset = self.load_translation_dataset('train', aug_path, lang_pair=p_pair, pair_score=True)
            data.append((f'{src}{tgt}-AUG', dataset))
            if args.augpara_reverse:
                logger.info(f'Reversing aug data: {p_pair} at {aug_path}')
                r_dataset = self.load_translation_dataset('train', aug_path, lang_pair=f'{tgt}-{src}', pair_score=True)
                data.append((f'{tgt}{src}-AUG', r_dataset))

        return RoundRobinZipDatasets(OrderedDict(data))

    def build_generator(self, models, args, eos=None, lang=None, **unused):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            )
        else:
            if eos is None:
                # likely python generate.py
                # rely on valid-lang-pairs to determine
                _src, _tgt = self.args.valid_lang_pairs.split(",")[0].split("-")
                lang = _tgt
                eos = self.target_dictionary.index("[{}]".format(_tgt))
                top_freq = getattr(self.args, "top_frequency", -1)
                logger.info(f'Inference mode from fairseq-py: {top_freq=} {eos} {lang}')

            rm_langs = [x for x in self.args.langs.split(",") if x != lang]
            rm_lang_indices = [self.dictionary.index(f"[{x}]") for x in rm_langs]
            logger.info(f'Build generator ({lang}, {eos}): {rm_lang_indices=}')
            return OBTmBARTSequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                eos=eos,
                top_frequency=getattr(self.args, "top_frequency", 0.99),
                top_freq_dict=self.freq_dicts[lang] if self.freq_dicts is not None else None,
                rm_lang_indices=rm_lang_indices,
            )

    def build_bt_generator(self, model, other_lang, **kwargs):
        eos = self.dictionary.index("[{}]".format(other_lang))
        rm_langs = [x for x in self.args.langs.split(",") if x != other_lang]
        rm_lang_indices = [self.dictionary.index(f"[{x}]") for x in rm_langs]
        logger.info(f'Build BT generator ({other_lang}, {eos}): {rm_lang_indices=}')
        generator = OBTmBARTSequenceGenerator(
            [model],
            tgt_dict=self.dictionary,
            beam_size=1,
            max_len_a=1.5,
            max_len_b=5,
            min_len=4,
            # keep 1 to be able to prepend bos
            max_len=model.max_decoder_positions() - 1,
            eos=eos,
            top_frequency=getattr(self.args, "top_frequency", 0.99),
            top_freq_dict=self.freq_dicts[other_lang] if self.freq_dicts is not None else None,
            rm_lang_indices=rm_lang_indices,
        )
        return generator

    # FINAL v3 back-translate_samples
    def backtranslate_sample(self, smp, orig_lang, other_lang, update_num) -> None:
        """
        * WARNING: smp is modified in place.
        * At the start of this function, `smp` has the same input and target:
          |---------------------------------------------------------------------------------------------|
          | smp['net_input']['src_tokens']    | smp['net_input']['prev_output_tokens'] | smp['target]   |
          | (from data) hello world </s> [en] | [en] hello world </s>                  | hello world </s> [en]
          |---------------------------------------------------------------------------------------------|

        * We call generator.generate(smp, bos_token = token("ro")),
        and copy the result as input
        * At the end, `smp` has the translation to other language.
          |---------------------------------------------------------------------------------------------|
          | smp['net_input']['src_tokens']    | smp['net_input']['prev_output_tokens'] | smp['target]   |
          | (from data) salut lume </s> [ro]  | [en] hello world </s>                  | hello world </s> [en]
          |---------------------------------------------------------------------------------------------|
        NOTE: make sure source has </s> token
        NOTE: Stricter version
        """
        prepend_bos = getattr(self.args, "prepend_bos", False)
        bos_token = _mbart_lang_token_index(self.dictionary, other_lang)
        eos = self.dictionary.eos()
        actual_bos = self.dictionary.bos()
        no_bt_strict = self.args.no_bt_strict

        generated = self.sequence_generators[orig_lang].generate(
            models=[], sample=smp, bos_token=bos_token,
            no_freq_mask=update_num > self.args.no_top_freq_after
        )
        if no_bt_strict:
            max_length = max([gn[0]["tokens"].size(0) for gn in generated])
            net_input = smp["net_input"]
            n_src_tokens = torch.empty(
                size=(len(generated), max_length), dtype=net_input["src_tokens"].dtype
            )
            n_src_lengths = torch.empty(
                len(generated), dtype=net_input["src_lengths"].dtype
            )
            for i, gn in enumerate(generated):
                tokens = gn[0]["tokens"]
                tokens_size = tokens.size(0)
                padding_needed = max_length - tokens_size
                tokens = F.pad(tokens, (0, padding_needed), value=self.dictionary.pad())
                n_src_tokens[i] = tokens
                n_src_lengths[i] = tokens_size
        else:
            max_length = max([gn[0]["tokens"].size(0) for gn in generated])
            net_input = smp["net_input"]
            gen_tokens = []
            for i, gn in enumerate(generated):
                tokens = gn[0]["tokens"]
                tokens_size = tokens.size(0)
                assert tokens[-1] == bos_token, f'!= bos {bos_token} {tokens}'
                if len(tokens) > 2 and tokens[-2] != eos:
                    tokens = torch.cat((tokens[:-2], torch.tensor([eos, bos_token]).to(tokens)), 0)
                if prepend_bos and tokens[0] != actual_bos:
                    tokens = torch.cat((torch.tensor([actual_bos]).to(tokens), tokens), 0)
                gen_tokens.append(tokens)
            max_length = max(x.size(0) for x in gen_tokens)
            n_src_tokens = torch.empty(
                size=(len(generated), max_length), dtype=net_input["src_tokens"].dtype
            )
            n_src_lengths = torch.empty(
                len(generated), dtype=net_input["src_lengths"].dtype
            )
            for i, x in enumerate(gen_tokens):
                tokens_size = x.size(0)
                padding_needed = max_length - tokens_size
                x = F.pad(x, (0, padding_needed), value=self.dictionary.pad())
                n_src_tokens[i] = x
                n_src_lengths[i] = tokens_size

        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = n_src_tokens.to(device)
        net_input["src_lengths"] = n_src_lengths.to(device)


@register_task("strtopfreqdy_augparascore_online_backtranslation_from_pretrained_bart")
class StrictTopFreqDynamicV4AugParaOnlineBackTranslationFromPretBARTTask(TopFreqDynamicV3AugParaOnlineBackTranslationFromPretBARTTask):
    """
    Same as TopFreqDynamicV3AugParaOnlineBackTranslationFromPretBARTTask
    - Strict back-translation process
    - Top-frequency built-in: during BT, only generate tokens of 99% percentile in the target corpus
    - Augmentation mined dataset integration with Dynamic lambda_aug and rank-based weighted XE
    """
    pass
