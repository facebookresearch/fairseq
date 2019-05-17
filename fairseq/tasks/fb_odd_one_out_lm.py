# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import numpy as np

from fairseq.data import (
    ConcatDataset,
    Dictionary,
    TokenBlockDataset,
    indexed_dataset
)
from fairseq.data.fb_odd_one_out_dataset import OddOneOutDataset

from .language_modeling import LanguageModelingTask
from . import register_task


@register_task('odd_one_out_lm')
class OddOneOutLMTask(LanguageModelingTask):
    """
    Train a language model with the odd-one-out prediction task.

    Currently supports only self-target models (i.e., bidirectional LM).

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.odd_one_out_lm_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-break-mode', choices=['eos'], default='eos'),
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--short-item-prob', default=0., type=float,
                            help='prob of returning a short item (1 or 2 sentences)')
        # fmt: on

    def __init__(self, args, dictionary):
        super().__init__(
            args, dictionary, output_dictionary=dictionary, targets=['self'],
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = None
        if args.data:
            paths = args.data.split(':')
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
            print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        loaded_datasets = []

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(data_path, split_k)
            ds = indexed_dataset.make_dataset(
                path,
                impl=self.args.dataset_impl,
                fix_lua_indexing=True,
                dictionary=self.dictionary
            )

            if ds is None:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            loaded_datasets.append(
                TokenBlockDataset(
                    ds, ds.sizes, self.args.tokens_per_sample,
                    pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                    break_mode='eos', include_targets=True,
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

        self.datasets[split] = OddOneOutDataset(
            dataset=dataset,
            sizes=sizes,
            vocab=self.dictionary,
            max_tokens=self.args.tokens_per_sample,
            short_item_prob=self.args.short_item_prob,
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        model.register_classification_head('ooo_head', num_classes=2)
        return model
