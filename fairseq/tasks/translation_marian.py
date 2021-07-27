# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from fairseq.data import (
    data_utils,
    TokenizerDictionary,
    RawLabelDataset,
    BertTokenizerDataset,
)
from fairseq.tasks import FairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task('translation_marian')
class SentencePredictionChineseBertTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')

    def __init__(self, args, data_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = args.max_source_positions
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        self.tokenizer = data_dictionary.tokenizer
        self.args = args

    @classmethod
    def load_dictionary(cls, args, model_path):
        dictionary = TokenizerDictionary.load(model_path)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            args.load_hf_bert_from,
        )

        return SentencePredictionChineseBertTask(args, data_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_raw_dataset(type):
            split_path = get_path(type, split)
            dataset = data_utils.load_indexed_raw_str_dataset(
                split_path,
            )
            return dataset

        input0 = make_raw_dataset('input0')
        assert input0 is not None, 'could not find dataset: {}'.format(get_path('input0', split))
        src_raw = input0
        tgt_raw = None
        label_path = "{0}.label".format(get_path('label', split))
        if os.path.exists(label_path):
            def parse_target(i, line):
                l = int(line.strip())
                return l

            with open(label_path) as h:
                tgt_raw = RawLabelDataset([
                    parse_target(i, line.strip())
                    for i, line in enumerate(h.readlines())
                ])

        self.datasets[split] = BertTokenizerDataset(src_raw, tgt_raw, self.tokenizer)
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        # logger.info("=" * 100 + " task.build_model " + "=" * 100)
        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, 'classification_head_name', 'sentence_classification_head'),
            num_classes=self.args.num_classes,
        )

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
