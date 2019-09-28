# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from torch.utils.data import ConcatDataset

from fairseq.data import (
    Dictionary, IndexedInMemoryDataset, IndexedRawTextDataset,
    data_utils,
)


from fairseq.data.shuffle_dataset import (
    ModifiedBlockPairDataset,
    ModifiedBertDataset,
)

from . import FairseqTask, register_task


class BertDictionary(Dictionary):
    """Dictionary for BERT tasks
        extended from Dictionary by adding support for cls as well as mask symbols"""
    def __init__(
        self,
        pad='[PAD]',
        unk='[UNK]',
        class_positive='[CLS]',
        class_negative='[MASK]',
        sep='[SEP]'
    ):
        super().__init__(pad, unk)
        (
            self.class_positive_word,
            self.class_negative_word,
            self.sep_word,
        ) = class_positive, class_negative, sep
        #self.class_positive_index = self.add_symbol(class_positive)
        #self.class_negative_index = self.add_symbol(class_negative)
        #self.sep_index = self.add_symbol(sep)
        self.nspecial = len(self.symbols)

    def class_positive(self):
        """Helper to get index of cls symbol"""
        idx = self.add_symbol(self.class_positive_word)
        #print (idx, self.class_positive_word)
        return idx

    def class_negative(self):
        """Helper to get index of cls symbol"""
        idx = self.add_symbol(self.class_negative_word)
        #print (idx, self.class_negative_word)
        return idx

    def sep(self):
        """Helper to get index of sep symbol"""
        idx = self.add_symbol(self.sep_word)
        #print (idx, self.sep_word)
        return idx


@register_task('shuffle_transformer_lm')
class ShuffleTransformerLMTask(FairseqTask):
    """
    Train BERT model.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments'
                                 ' per sample for BERT dataset')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--break-mode', default="sentence", type=str, help='mode for breaking sentence')
        parser.add_argument('--short-seq-prob', default=0.1, type=float)
        parser.add_argument('--shuffle-instance', default=False, action='store_true')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = dictionary
        self.seed = args.seed

    @property
    def target_dictionary(self):
        return self.dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))
        if not hasattr(args, 'shuffle_instance'):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(self.args.data, split_k)

            if self.args.raw_text and IndexedRawTextDataset.exists(path):
                ds = IndexedRawTextDataset(path, self.dictionary)
                tokens = [t for l in ds.tokens_list for t in l]
            elif not self.args.raw_text and IndexedInMemoryDataset.exists(path):
                ds = IndexedInMemoryDataset(path, fix_lua_indexing=False)
                tokens = ds.buffer
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))
            with data_utils.numpy_seed(self.seed + k):
                loaded_datasets.append(
                    ModifiedBlockPairDataset(
                        tokens,
                        ds.sizes,
                        self.args.tokens_per_sample,
                        pad=self.dictionary.pad(),
                        class_positive=self.dictionary.class_positive(),
                        class_negative=self.dictionary.class_negative(),
                        sep=self.dictionary.sep(),
                        vocab = self.dictionary,
                        break_mode=self.args.break_mode,
                        short_seq_prob=self.args.short_seq_prob
                    ))

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[-1])))

            if not combine:
                break

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])
  
        self.datasets[split] = ModifiedBertDataset(
            dataset, sizes, self.dictionary,
            shuffle=self.args.shuffle_instance, seed=self.seed,
        )
