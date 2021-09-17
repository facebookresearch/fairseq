import logging
import os
import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    ConcatDataset,
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
)

from .swav_task_utils import VanillaSwavBaseTaskWrapper

from fairseq.tasks import register_task
from fairseq.tasks.multilingual_masked_lm import MultiLingualMaskedLMTask

from ..data.swav_dataset import SwavExtrapolatePadDataset, SwavExtrapolateNumelDataset
from ..data.swav_dataset import SwavExtrapolateLangIdDataset

logger = logging.getLogger(__name__)


@register_task("multilingual_swav_lm")
class MultiLingualSwavLMTask(MultiLingualMaskedLMTask, VanillaSwavBaseTaskWrapper):
    """
    MultiLingualMaskedLMTask with SwavLoss
    """
    @staticmethod
    def add_args(parser):
        MultiLingualMaskedLMTask.add_args(parser=parser)
        VanillaSwavBaseTaskWrapper.add_swav_args(parser)
    
    @property
    def swav_prepend_token(self):
        prepend_tok = self.source_dictionary.eos() if getattr(self.args, 'prepend_eos', False) else self.source_dictionary.bos()
        return prepend_tok

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
        return loss, sample_size, logging_output

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
                })
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
    
    def build_para_multilingual_dataset(self, languages, lang_datasets, split, epoch):
        """
        MultiLingualSwavLMTask sample regardless of languages, like MultiLingualMaskedLMTask
            - restricted version must sample same no. of data per language in a batch
        *** Expect the datasets to be exact parallel data
        """
        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        assert all([len(d) == len(lang_datasets[0]) for d in lang_datasets[1:]]), f'inconsistent lengths {lang_datasets}'
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
                })
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
            "Language to id mapping: {}".format({lang: id for id, lang in enumerate(languages)})
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
            if not self.args.no_token_block:
                # create continuous blocks of tokens
                dataset = TokenBlockDataset(
                    dataset,
                    dataset.sizes,
                    self.args.tokens_per_sample - 1,  # one less for <s>
                    pad=self.source_dictionary.pad(),
                    eos=self.source_dictionary.eos(),
                    break_mode=self.args.sample_break_mode,
                )
            logger.info("loaded {} blocks from: {}, notokblock: {}".format(
                len(dataset), split_path, self.args.no_token_block))

            # create noise dataset
            src_noise_dataset = self.create_swav_noising_dataset(dataset, self.source_dictionary)

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.swav_prepend_token)

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
                        "src_langs": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                    },
                    "net_swav_input": {
                        "src_tokens": SwavExtrapolatePadDataset(
                            src_noise_dataset,
                            pad_idx=self.source_dictionary.pad(), left_pad=False
                        ),
                        "src_lengths": SwavExtrapolateNumelDataset(src_noise_dataset, reduce=False),
                        "src_langs": SwavExtrapolateLangIdDataset(src_noise_dataset, lang_id, reduce=False),
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

    def load_dataset_for_analysis(self, split, epoch=1, combine=False, **kwargs):
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
            "Language to id mapping: {}".format({lang: id for id, lang in enumerate(languages)})
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
            if not self.args.no_token_block:
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
            dataset = PrependTokenDataset(dataset, self.swav_prepend_token)

            src_dataset = dataset

            lang_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                        "src_langs": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                    },
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                    "lang_id": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                    "id": RawLabelDataset(np.arange(src_dataset.sizes.shape[0])),
                },
                sizes=[src_dataset.sizes],
            )
            lang_datasets.append(lang_dataset)
                
        self.build_multilingual_dataset(
            languages, lang_datasets, split, epoch, 
            train_lang_sep=kwargs.get('train_lang_sep', False)
        )
    
    def load_dataset_for_para_analysis(self, split, epoch=1, combine=False, **kwargs):
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
            "Language to id mapping: {}".format({lang: id for id, lang in enumerate(languages)})
        )
        # expect the data files are well aligned!!
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
            if not self.args.no_token_block:
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
            dataset = PrependTokenDataset(dataset, self.swav_prepend_token)

            src_dataset = dataset

            lang_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                        "src_langs": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                    },
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                    "lang_id": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                },
                sizes=[src_dataset.sizes],
            )
            lang_datasets.append(lang_dataset)
                
        self.build_multilingual_dataset(languages, lang_datasets, split, epoch)
    
