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
    MonolingualDataset, TokenBlockDataset,
)

from . import FairseqTask, register_task


@register_task('language_modeling')
class LanguageModelingTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--sample-break-mode', metavar='VAL',
                            choices=['none', 'complete', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int, metavar='N',
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--right-to-left', default=False, action='store_true',
                            help='if set, trains a language model right-to-left (instead of left-to-right)')

    def __init__(self, args, dictionary):
        super().__init__(args)

        args.right_to_left = getattr(args, 'right_to_left', False)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False):
        """Load a dataset split."""

        loaded_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(self.args.data, split_k)

            if self.args.raw_text and IndexedRawTextDataset.exists(path):
                ds = IndexedRawTextDataset(path, self.dictionary)
                tokens = [t for l in ds.tokens_list for t in l]
            elif not self.args.raw_text and IndexedInMemoryDataset.exists(path):
                ds = IndexedInMemoryDataset(path, fix_lua_indexing=True)
                tokens = ds.buffer
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

            loaded_datasets.append(
                TokenBlockDataset(
                    tokens, ds.sizes, self.args.tokens_per_sample, self.args.sample_break_mode,
                    include_targets=True, reverse=self.args.right_to_left,
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

        self.datasets[split] = MonolingualDataset(dataset, sizes, self.dictionary, shuffle=False)

    @property
    def target_dictionary(self):
        return self.dictionary
