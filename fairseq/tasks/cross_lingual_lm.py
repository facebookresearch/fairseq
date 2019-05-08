# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

from collections import OrderedDict

import numpy as np

from fairseq import tokenizer
from fairseq.data.masked_lm_dictionary import MaskedLMDictionary

from fairseq.data import (
    ConcatDataset,
    indexed_dataset,
    TokenBlockDataset,
)

from fairseq.data import Dictionary
from fairseq.data.masked_lm_dataset import MaskedLMDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset

from . import FairseqTask, register_task


@register_task('cross_lingual_lm')
class CrossLingualLMTask(FairseqTask):
    """
    Task for training cross-lingual language models.
    For more details look at: https://arxiv.org/pdf/1901.07291.pdf
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments'
                                 ' per sample')
        parser.add_argument('--monolingual-langs', default='en', type=str,
                            help='comma separated list of languages for which we'
                                 ' want to train XLM on')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--shuffle', action='store_true',
                            help='shuffle each monolingual dataset while'
                            ' training')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.distributed_world_size = args.distributed_world_size
        self.langs2id = self._lang_to_id(args.monolingual_langs)
        self.default_key = None

    def _lang_to_id(
            self,
            languages: str
    ):
        """
        Build a map from languages to ids. These ids are used as segment labels
        for cross-lingual LM training.
        """
        lang2id = {}
        langs = [l.strip() for l in languages.split(',')]
        for id, lang in enumerate(langs):
            lang2id[lang] = id
        return lang2id

    @classmethod
    def load_dictionary(cls, filename):
        return MaskedLMDictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = MaskedLMDictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @property
    def target_dictionary(self):
        return self.dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        dictionary = MaskedLMDictionary.load(os.path.join(args.data, 'dict.txt'))

        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def _load_single_lang_dataset(self, split, epoch):
        loaded_datasets = []

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(data_path, split_k)

            ds = indexed_dataset.make_dataset(
                path, impl=self.args.dataset_impl, fix_lua_indexing=True,
                dictionary=self.dictionary,
            )
            if ds is None:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            # Since we append each block with the classification_token,
            # we need to effectively create blocks of length
            # tokens_per_sample-1
            loaded_datasets.append(
                TokenBlockDataset(
                    ds, ds.sizes, self.args.tokens_per_sample - 1,
                    pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                )
            )

            print('| {} {} {} examples'.format(data_path, split_k, len(loaded_datasets[-1])))

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])

        return dataset, sizes

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        dataset_map = OrderedDict()

        for lang in self.langs2id.keys():
            if self.default_key is None:
                self.default_key = lang

            # Datasets are expected to be in "split.lang" format (Eg: train.en)
            language_split = '{}.{}'.format(split, lang)

            block_dataset, sizes = self._load_single_lang_dataset(split=language_split, epoch=epoch)

            dataset_map[lang] = MaskedLMDataset(
                dataset=block_dataset,
                sizes=sizes,
                vocab=self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.dictionary.mask(),
                classif_token_idx=self.dictionary.eos(),
                sep_token_idx=self.dictionary.eos(),
                shuffle=getattr(self.args, 'shuffle', False),
                has_pairs=False,
                segment_id=self.langs2id[lang],
                seed=self.seed,
            )

        self.datasets[split] = MultiCorpusSampledDataset(
            dataset_map, default_key=self.default_key
        )
        print('| {} {} {} examples'.format(
            self.args.data.split(':')[epoch], split, len(self.datasets[split])
            )
        )
