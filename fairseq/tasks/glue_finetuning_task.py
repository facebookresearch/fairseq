# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

from . import FairseqTask, register_task

from fairseq.data import (
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    MonolingualDataset,
    MonolingualDatasetsWithLabels
)
from fairseq.data.masked_lm_dictionary import BertDictionary


@register_task('glue_finetune')
class GlueFinetuneTask(FairseqTask):
    """
    Task for finetuning glue task (either sentence or pair classification)
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--max-positions', type=int, default=512,
                            help='max input length')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>)')
        parser.add_argument('--add-segment-embed', action='store_true',
                            help='add segment embeddings')

        parser.add_argument('--separator-token', action='store_true',
                            help='add separator token between inputs')

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.tokens_per_sample = args.max_positions
        dictionary = BertDictionary.load(os.path.join(args.data, 'input0', 'dict.txt'))
        print('| [input] dictionary: {} types'.format(len(dictionary)))
        return GlueFinetuneTask(args, dictionary)

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

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type):
            return os.path.join(self.args.data, type, split)

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        input_datasets = []

        for i in itertools.count():
            input_type = 'input{0}'.format(i)
            input_dataset = indexed_dataset(get_path(input_type), self.dictionary)
            # we require at least one input dataset, others are optional.
            if i == 0 and input_dataset is None:
                raise FileNotFoundError('Dataset not found: ' + get_path(input_type))

            if input_dataset is not None:
                # wrap input dataset(s) in MonolingualDataset
                input_datasets.append(
                    MonolingualDataset(
                        input_dataset,
                        input_dataset.sizes,
                        self.dictionary,
                        self.dictionary,
                        add_eos_for_other_targets=False,
                        shuffle=False,
                        add_bos_token=False,
                    )
                )
            else:
                break

        # labels are also optional
        label_file_path = '{0}.label'.format(get_path('label'))
        labels = []
        with open(label_file_path) as fin:
            for line in fin:
                labels.append(int(line.strip()))

        if self.args.separator_token:
            separator_token = self.dictionary.sep()
            assert separator_token != self.dictionary.unk()
        else:
            separator_token = None

        self.datasets[split] = MonolingualDatasetsWithLabels(
            input_datasets=input_datasets,
            labels=labels,
            shuffle=True,
            add_bos_token=self.args.add_bos_token,
            concat_inputs=True,
            separator_token=separator_token,
            cls_token=self.dictionary.cls(),
        )

        print('| {} {} {} examples'.format(self.args.data, split, len(self.datasets[split])))
