import logging
import os
import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
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
)

from .swav_task_utils import VanillaSwavBaseTaskWrapper
from fairseq.data.append_token_dataset import AppendTokenDataset
from fairseq.tasks.multilingual_denoising import MultilingualDenoisingTask
from fairseq.tasks import register_task
from fairseq.tasks.masked_lm import get_whole_word_mask
from ..data.swav_dataset import SwavExtrapolateDenoisingDataset


logger = logging.getLogger(__name__)


@register_task("multilingual_swav_denoising")
class MultilingualSwavDenoisingTask(MultilingualDenoisingTask, VanillaSwavBaseTaskWrapper):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.SHOW_SAMPLES_INTERVAL = 10000
        self._show_samples_ctr = self.SHOW_SAMPLES_INTERVAL
        self.SHOW_SAMPLES_NUMBER = 5

    @staticmethod
    def add_args(parser):
        MultilingualDenoisingTask.add_args(parser)
        VanillaSwavBaseTaskWrapper.add_swav_args(parser)
        parser.add_argument("--replace-eos", default=False, action="store_true",
                            help="Replace eos token with lang_id")
        parser.add_argument("--no-bos", default=False, action="store_true",
                            help="Replace eos token with lang_id")
        parser.add_argument("--display-samples", default=False, action="store_true",
                            help="display some samples to confirm")

    def display_samples_once_in_a_while(self, smp):
        if 1 < self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr += 1
            return
        elif self._show_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr = 0
        else:
            self._show_samples_ctr += 1

        ignores_syms = [self.dictionary.pad()]

        ln = smp["net_input"]["src_tokens"].shape[0]
        bpe_symbol = "sentencepiece"

        logger.info(f"(r:{self.args.distributed_rank}) : {ln} denoising samples")
        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            src_len = smp['net_input']['src_lengths'][i]
            src_lang = self.dictionary.string(src_tokens[src_len - 1:src_len], extra_symbols_to_ignore=ignores_syms)
            src_str = self.dictionary.string(src_tokens, bpe_symbol, extra_symbols_to_ignore=ignores_syms)
            logger.info(f"\n{i}\t{src_lang}  {src_str}\n")

        if "net_swav_input" in smp:
            swln = smp["net_swav_input"]["src_tokens"].shape[0]
            logger.info(f"(r:{self.args.distributed_rank}) : {swln} SWAV denoising samples")
            for i in range(min(swln, self.SHOW_SAMPLES_NUMBER)):
                src_tokens = smp["net_swav_input"]["src_tokens"][i]
                src_len = smp['net_swav_input']['src_lengths'][i]
                src_lang = self.dictionary.string(src_tokens[src_len - 1:src_len], extra_symbols_to_ignore=ignores_syms)
                src_str = self.dictionary.string(src_tokens, bpe_symbol, extra_symbols_to_ignore=ignores_syms)
                logger.info(f"\n{i}\tSWAV-{src_lang}  {src_str}\n")

    def _default_train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
            if update_num < self.args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
        return loss, sample_size, logging_output

    def _subsequent_train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad):
        model.train()
        model.set_num_updates(update_num)
        logging_output = None
        lm_loss = 0
        if self.args.swav_lambda < 1.0:
            # disable mlm in case of lambda == 1
            # lm loss
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, sample, mode='mlm')
                loss *= (1.0 - criterion.swav_lambda)
                if ignore_grad:
                    loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
                lm_loss = loss.detach().item()
                del loss
        # swav loss
        with torch.autograd.profiler.record_function("forward"):
            swav_loss, swav_sample_size, swav_logging_output = criterion(model, sample, mode='swav')
            if logging_output is None:
                logging_output, sample_size = swav_logging_output, swav_sample_size
            else:
                logging_output['swav_loss'] = swav_logging_output['loss']
            if ignore_grad:
                swav_loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(swav_loss)
            swav_loss_d = swav_loss.detach().item()
            del swav_loss
        if update_num < self.args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # combine
        agg_loss = lm_loss + swav_loss_d
        return agg_loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad):
        # consecutively compute the lm and swav loss, not in combination due to memory issue
        if self.args.subsequent_loss:
            loss, sample_size, logging_output = self._subsequent_train_step(
                sample, model, criterion, optimizer, update_num, ignore_grad
            )
        else:
            loss, sample_size, logging_output = self._default_train_step(
                sample, model, criterion, optimizer, update_num, ignore_grad
            )
            if self.args.display_samples:
                self.display_samples_once_in_a_while(sample)
        return loss, sample_size, logging_output

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        paths = args.data.split(":")
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))

        data_path = paths[0]
        if args.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = args.langs.split(",")

        if args.add_lang_token:
            for lang in languages:
                dictionary.add_symbol("[{}]".format(lang))

        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def create_swav_denoising_dataset(self, language, lang_id, dataset, dictionary, lang_mask_whole_words, **kwargs):
        dataset = SwavExtrapolateDenoisingDataset(
            lang_id,
            dataset,
            dataset.sizes,
            dictionary,
            self.mask_idx,
            mask_whole_words=lang_mask_whole_words,
            shuffle=self.args.shuffle_instance,
            seed=self.seed,
            args=self.args,
            rand_factor=self.args.rand_factor,
            eos=None if not self.args.add_lang_token else self.source_dictionary.index(
                "[{}]".format(language)),
        )
        return dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(":")
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
            data_languages = languages
        else:
            languages = self.langs.split(",")
            data_languages = []
            missing_languages = []
            for name in languages:
                p = os.path.join(data_path, name)
                if not os.path.exists(p):
                    missing_languages.append(name)
                else:
                    data_languages.append(name)
            logger.warning(f'data language not found: {missing_languages}')
        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: {}".format({lang: id for id, lang in enumerate(languages)})
        )

        mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(",")
        lang_datasets = []
        for lang_id, language in enumerate(languages):
            if language not in data_languages:
                continue
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

            end_token = (
                self.source_dictionary.index("[{}]".format(language))
                if self.args.add_lang_token
                else self.source_dictionary.eos()
            )

            # create continuous blocks of tokens
            if not self.args.no_token_block:
                logger.warning('Use TokenBlockDataset')
                dataset = TokenBlockDataset(
                    dataset,
                    dataset.sizes,
                    self.args.tokens_per_sample - 2,  # one less for <s>
                    pad=self.source_dictionary.pad(),
                    eos=end_token,
                    break_mode=self.args.sample_break_mode,
                )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            if not self.args.no_bos:
                dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
            else:
                assert not hasattr(self.args, 'prot_hidden') or self.args.prot_hidden != "bos"
                logger.info(f'{self.args.no_bos=}, {hasattr(self.args, "prot_hidden")=}, {getattr(self.args, "prot_hidden", "bos")}')
            dataset = AppendTokenDataset(dataset, end_token)

            lang_mask_whole_words = (
                mask_whole_words
                if language not in language_without_segmentations
                else None
            )
            lang_dataset = self.create_swav_denoising_dataset(
                language, lang_id, dataset, self.dictionary,
                lang_mask_whole_words=lang_mask_whole_words
            )
            lang_datasets.append(lang_dataset)
        self.build_multilingual_dataset(data_languages, lang_datasets, split, epoch)

    def load_dataset_for_analysis(self, split, epoch=1, combine=False, **kwargs):
        paths = self.args.data.split(":")
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
            data_languages = languages
        else:
            languages = self.langs.split(",")
            data_languages = []
            missing_languages = []
            for name in languages:
                p = os.path.join(data_path, name)
                if not os.path.exists(p):
                    missing_languages.append(name)
                else:
                    data_languages.append(name)
            logger.warning(f'data language not found: {missing_languages}')

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: {}".format({lang: id for id, lang in enumerate(languages)})
        )

        lang_datasets = []
        for lang_id, language in enumerate(languages):
            if language not in data_languages:
                continue
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

            end_token = (
                self.source_dictionary.index("[{}]".format(language))
                if self.args.add_lang_token
                else self.source_dictionary.eos()
            )

            # create continuous blocks of tokens
            if not self.args.no_token_block:
                logger.warning('Use TokenBlockDataset')
                dataset = TokenBlockDataset(
                    dataset,
                    dataset.sizes,
                    self.args.tokens_per_sample - 2,  # one less for <s>
                    pad=self.source_dictionary.pad(),
                    eos=end_token,
                    break_mode=self.args.sample_break_mode,
                )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            if not self.args.no_bos:
                dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
            else:
                assert not hasattr(self.args, 'prot_hidden') or self.args.prot_hidden != "bos"
                logger.info(f'{self.args.no_bos=}, {hasattr(self.args, "prot_hidden")=}, {getattr(self.args, "prot_hidden", "bos")}')
            dataset = AppendTokenDataset(dataset, end_token)

            lang_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": PadDataset(
                            dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(dataset, reduce=False),
                        "src_langs": RawLabelDataset([lang_id] * dataset.sizes.shape[0]),
                    },
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(dataset, reduce=True),
                    "lang_id": RawLabelDataset([lang_id] * dataset.sizes.shape[0]),
                    "id": RawLabelDataset(np.arange(dataset.sizes.shape[0])),
                },
                sizes=[dataset.sizes],
            )
            lang_datasets.append(lang_dataset)
        self.build_multilingual_dataset(data_languages, lang_datasets, split, epoch, train_lang_sep=kwargs.get('train_lang_sep', False))

    def build_multilingual_dataset(self, languages, lang_datasets, split, epoch, train_lang_sep=False):
        """
        MultiLingualSwavLMTask sample regardless of languages, like MultiLingualMaskedLMTask
            - restricted version must sample same no. of data per language in a batch
        """
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
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: {}".format({
                    lang: "{0:.4f}".format(sample_probs[id])
                    for id, lang in enumerate(languages)
                }),
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: {}".format({
                    lang: "{0:.2f}".format(size_ratio[id])
                    for id, lang in enumerate(languages)
                }),
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
                    self.datasets[f'{split}_{l}'] = d
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
