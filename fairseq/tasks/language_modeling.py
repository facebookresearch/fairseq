# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import torch
import numpy as np

from fairseq.data import (
    ConcatDataset,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    MonolingualDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)

from . import FairseqTask, register_task


@register_task('language_modeling')
class LanguageModelingTask(FairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-break-mode',
                            choices=['none', 'complete', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        # fmt: on

    def __init__(self, args, dictionary, output_dictionary, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary

        if targets is None:
            targets = ['future']
        self.targets = targets

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = None
        output_dictionary = None
        if args.data:
            dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)

        # upgrade old checkpoints
        if hasattr(args, 'exclude_self_target'):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, 'self_target', False):
            targets.append('self')
        if getattr(args, 'future_target', False):
            targets.append('future')
        if getattr(args, 'past_target', False):
            targets.append('past')
        if len(targets) == 0:
            # standard language modeling
            targets = ['future']

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)

        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError('Unsupported language modeling target: {}'.format(target))

        return model

    def load_dataset(self, split, combine=False, **kwargs):
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
            elif not self.args.raw_text and IndexedDataset.exists(path):
                if self.args.lazy_load:
                    ds = IndexedDataset(path, fix_lua_indexing=True)
                else:
                    ds = IndexedCachedDataset(path, fix_lua_indexing=True)
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

            loaded_datasets.append(
                TokenBlockDataset(
                    ds, ds.sizes, self.args.tokens_per_sample,
                    pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                    break_mode=self.args.sample_break_mode, include_targets=True,
                )
            )

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[-1])))

            if not combine:
                break

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])

        add_eos_for_other_targets = self.args.sample_break_mode is not None and self.args.sample_break_mode != 'none'

        self.datasets[split] = MonolingualDataset(
            dataset, sizes, self.dictionary, self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets, shuffle=True,
            targets=self.targets,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return TransformEosDataset(
            MonolingualDataset(
                TokenBlockDataset(
                    src_tokens,
                    src_lengths,
                    block_size=None,
                    pad=self.source_dictionary.pad(),
                    eos=self.source_dictionary.eos(),
                    break_mode='eos',
                    include_targets=False,
                ),
                src_lengths,
                self.source_dictionary,
                self.target_dictionary,
                add_eos_for_other_targets=False,
                shuffle=False,
            ),
            eos=self.source_dictionary.eos(),
            # remove EOS since this will be used as a prefix for generation
            remove_eos_from_src=True,
            has_target=False,
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None:
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample['net_input']['src_tokens']
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
