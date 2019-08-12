# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
)

from . import FairseqTask, register_task


@register_task('sentence_ranking')
class SentenceRankingTask(FairseqTask):
    """
    Ranking task on multiple sentences.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=2,
                            help='number of sentences to be ranked')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='Truncate sequence to max_sequence_length')
        parser.add_argument('--max-option-length', type=int, default=None,
                            help='max length for each option')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.criterion == 'sentence_ranking', \
            'Must set --criterion=sentence_ranking'

        args.tokens_per_sample = args.max_positions

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'input0', 'dict.txt'),
            source=True,
        )
        print('| [input] dictionary: {} types'.format(len(data_dict)))
        return SentenceRankingTask(args, data_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset('input0', self.source_dictionary)
        input_options = [
            make_dataset(
                'input{idx}'.format(idx=idx + 1),
                self.source_dictionary
            )
            for idx in range(self.args.num_classes)
        ]

        if self.args.separator_token is not None:
            input0 = PrependTokenDataset(input0, self.args.separator_token)

        src_tokens = []
        for input_option in input_options:
            if self.args.init_token is not None:
                input_option = PrependTokenDataset(input_option, self.args.init_token)
            input_option = TruncateDataset(input_option, self.args.max_option_length)
            src_token = ConcatSentencesDataset(input_option, input0)
            if self.args.truncate_sequence:
                src_token = TruncateDataset(src_token, self.args.max_positions)
            src_tokens.append(src_token)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens[0]))

        dataset = {
            'id': IdDataset(),
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens[0], reduce=True),
        }

        for src_token_idx in range(len(src_tokens)):
            dataset.update(
                {
                    'net_input{idx}'.format(idx=src_token_idx+1): {
                        'src_tokens': RightPadDataset(
                            src_tokens[src_token_idx],
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        'src_lengths': NumelDataset(src_tokens[src_token_idx], reduce=False),
                    }
                }
            )

        label_path = '{}.label'.format(get_path('label', split))
        if os.path.exists(label_path):
            dataset.update(
                target=RawLabelDataset([
                    int(x.strip()) for x in open(label_path).readlines()
                ])
            )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[np.maximum.reduce([src_token.sizes for src_token in src_tokens])],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        print("| Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_classification_head(
            'sentence_classification_head',
            num_classes=1,
        )

        return model

    def max_positions(self):
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
