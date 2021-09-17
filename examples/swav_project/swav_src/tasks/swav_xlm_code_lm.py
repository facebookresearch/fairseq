import logging
import os
import numpy as np
from fairseq import utils
from fairseq.data import (
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    RawLabelDataset,
    data_utils,
)

from .swav_task_utils import VanillaSwavBaseTaskWrapper
from fairseq.tasks import register_task
from .swav_lm import MultiLingualSwavLMTask
from ..data.swav_dataset import SwavExtrapolateNoisingDataset, SwavExtrapolatePadDataset, SwavExtrapolateNumelDataset
from ..data.swav_dataset import SwavExtrapolatePrependTokenDataset, SwavExtrapolateLangIdDataset
from .xlm_code_tasks import MassMaskExpandDataset, MultilingualMassXLMTask, xlm_lm_setup_task

logger = logging.getLogger(__name__)


@register_task("multilingual_swav_lm_xlm")
class MultiLingualSwavLMXLMTask(MultiLingualSwavLMTask):
    """
    Multilingual LM task that try to fit XLM codebase
    To match XLM code
    * sentence based, not token_block_dataset
    * prepend and append eos to both sentences,
        * translation data (fi_xlm_preprocess.py) already append eos
        * only prepend
    """
    @classmethod
    def setup_task(cls, args, **kwargs):
        return xlm_lm_setup_task(cls, args, **kwargs)
    
    @property
    def swav_prepend_token(self):
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

            # create continuous blocks of tokens
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

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
                    "id": RawLabelDataset(np.arange(src_dataset.sizes.shape[0])),
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

        # mask_whole_words = self._get_whole_word_mask()
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
        self.build_multilingual_dataset(languages, lang_datasets, split, epoch, train_lang_sep=kwargs.get('train_lang_sep', False))


@register_task("multilingual_swav_mass_xlm")
class MultilingualSwavMassXLMTask(MultilingualMassXLMTask, VanillaSwavBaseTaskWrapper):
    """
    Mass training that fits with XLM codebase
        https://github.com/microsoft/MASS/tree/master/MASS-unsupNMT
    """
    
    @staticmethod
    def add_args(parser):
        MultilingualMassXLMTask.add_args(parser=parser)
        VanillaSwavBaseTaskWrapper.add_swav_args(parser)
        parser.add_argument("--aly-input-noise", default=False, action='store_true',
                            help="for load data analysis: input noise, it is being noised by default")
        parser.add_argument("--swav-prev", default=None, type=str,
                            help="set prev_output_tokens to net_swav_input")

    @property
    def swav_prepend_token(self):
        prepend_tok = self.source_dictionary.eos()
        return prepend_tok
    
    def create_swav_prev_datsets(self, index_dataset, noise_datset, lang_id, **kwargs):
        swav_prev = self.args.swav_prev
        if swav_prev is None or swav_prev == 'default':
            return {}
        elif swav_prev == 'noise_same':
            return {
                "prev_output_tokens": SwavExtrapolatePadDataset(
                    noise_datset,
                    pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
                "tgt_lengths": SwavExtrapolateNumelDataset(noise_datset, reduce=False),
                "tgt_langs": SwavExtrapolateLangIdDataset(noise_datset, lang_id, reduce=False),
            }
        elif swav_prev == 'noise':
            src_noise_dataset = self.create_swav_noising_dataset(index_dataset, self.source_dictionary, seed=self.args.seed + 1)
            return {
                "prev_output_tokens": SwavExtrapolatePadDataset(
                    src_noise_dataset,
                    pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
                "tgt_lengths": SwavExtrapolateNumelDataset(src_noise_dataset, reduce=False),
                "tgt_langs": SwavExtrapolateLangIdDataset(src_noise_dataset, lang_id, reduce=False),
            }
        elif swav_prev == "ori":
            dup_dataset = SwavExtrapolateNoisingDataset(
                index_dataset, self.source_dictionary, 
                seed=kwargs.get('seed', self.args.seed), 
                rand_factor=kwargs.get('rand_factor', self.args.rand_factor),
                noiser="placeholder",
                nonoise_duplicate=True,
            )
            if self.swav_prepend_token is not None:
                dup_dataset = SwavExtrapolatePrependTokenDataset(dup_dataset, self.swav_prepend_token)
            return {
                "prev_output_tokens": SwavExtrapolatePadDataset(
                    dup_dataset,
                    pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
                "tgt_lengths": SwavExtrapolateNumelDataset(dup_dataset, reduce=False),
                "tgt_langs": SwavExtrapolateLangIdDataset(dup_dataset, lang_id, reduce=False),
            }
        else:
            raise NotImplementedError(f'swav_prev: {swav_prev}')

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

            # create noise dataset
            index_dataset = dataset
            src_noise_dataset = self.create_swav_noising_dataset(dataset, self.source_dictionary)

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            # NOTE: for xlm, prepend eos not, bos
            dataset = PrependTokenDataset(dataset, self.swav_prepend_token)

            data_dict = {
                "net_input": {
                    "src_tokens": PadDataset(
                        dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(dataset, reduce=False),
                    "src_langs": RawLabelDataset([lang_id] * dataset.sizes.shape[0]),
                },
                "net_swav_input": {
                    "src_tokens": SwavExtrapolatePadDataset(
                        src_noise_dataset,
                        pad_idx=self.source_dictionary.pad(), left_pad=False
                    ),
                    "src_lengths": SwavExtrapolateNumelDataset(src_noise_dataset, reduce=False),
                    "src_langs": SwavExtrapolateLangIdDataset(src_noise_dataset, lang_id, reduce=False),
                },
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(dataset, reduce=True),
                "lang_id": RawLabelDataset([lang_id] * dataset.sizes.shape[0]),
                "id": RawLabelDataset(np.arange(dataset.sizes.shape[0])),
            }
            swav_prev = self.create_swav_prev_datsets(index_dataset, src_noise_dataset, lang_id)
            for k, v in swav_prev.items():
                data_dict['net_swav_input'][k] = v

            lang_dataset = MassMaskExpandDataset(
                data_dict,
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

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.swav_prepend_token)

            lang_dataset = MassMaskExpandDataset(
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
                vocab=self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                sizes=[dataset.sizes],
                seed=self.args.seed,
                span_len=self.args.span_len,
                word_mass=self.args.word_mass,
                no_input_noise=not self.args.aly_input_noise,
            )
            lang_datasets.append(lang_dataset)
                
        self.build_multilingual_dataset(
            languages, lang_datasets, split, epoch, 
            train_lang_sep=kwargs.get('train_lang_sep', False)
        )
    




