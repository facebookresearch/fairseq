# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from fairseq import tokenizer
from fairseq.data import (
    ConcatDataset,
    indexed_dataset,
    data_utils,
)

from fairseq.data import Dictionary
from fairseq.data.block_pair_dataset import BlockPairDataset
from fairseq.data.masked_lm_dataset import MaskedLMDataset
from fairseq.data.masked_lm_dictionary import BertDictionary

from . import FairseqTask, register_task


@register_task('masked_lm')
class MaskedLMTask(FairseqTask):
    """
    Task for training Masked LM (BERT) model.
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
                                 ' per sample for BERT dataset')
        parser.add_argument('--break-mode', default="doc", type=str, help='mode for breaking sentence')
        parser.add_argument('--shuffle-dataset', action='store_true', default=False)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = BertDictionary()
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
        paths = args.data.split(':')
        assert len(paths) > 0
        dictionary = BertDictionary.load(os.path.join(paths[0], 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        loaded_datasets = []

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        print("| data_path", data_path)

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(data_path, split_k)
            ds = indexed_dataset.make_dataset(
                path,
                impl=self.args.dataset_impl,
                fix_lua_indexing=True,
                dictionary=self.dictionary,
            )

            if ds is None:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            with data_utils.numpy_seed(self.seed + k):
                loaded_datasets.append(
                    BlockPairDataset(
                        ds,
                        self.dictionary,
                        ds.sizes,
                        self.args.tokens_per_sample,
                        break_mode=self.args.break_mode,
                        doc_break_size=1,
                    )
                )

            print('| {} {} {} examples'.format(data_path, split_k, len(loaded_datasets[-1])))

            if not combine:
                break

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])

        self.datasets[split] = MaskedLMDataset(
            dataset=dataset,
            sizes=sizes,
            vocab=self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.dictionary.mask(),
            classif_token_idx=self.dictionary.cls(),
            sep_token_idx=self.dictionary.sep(),
            shuffle=self.args.shuffle_dataset,
            seed=self.seed,
        )
