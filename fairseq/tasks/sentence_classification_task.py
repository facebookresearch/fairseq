# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from functools import reduce
import itertools
import numpy as np
import os

from torch.utils.data import ConcatDataset

from fairseq.data import (
    Dictionary, IndexedInMemoryDataset, IndexedRawTextDataset,
    SentenceClassificationDataset, TokenBlockDataset
)
from fairseq.meters import MCCMeter, AccuracyMeter

from . import FairseqTask, register_task


@register_task('sentence_classification')
class SentenceClassificationTask(FairseqTask):
    """
    Classify a sentence

    Args:
        dictionary (Dictionary): the dictionary for the input of the classifier

    The sentence classification task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.sentence_classification_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--num-labels', type=int, default=2,
                            help='number of labels')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.padding_idx = -100
        self.num_labels = args.num_labels

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = []
        loaded_labels = []

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
                    tokens, ds.sizes, 0, pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                    break_mode='eos', include_targets=False,
                ))

            with open(path + '.lbl', 'r') as lbl_f:
                lines = lbl_f.readlines()
                loaded_labels.extend(int(l) for l in lines)

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[-1])))

            if not combine:
                break

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])

        self.datasets[split] = SentenceClassificationDataset(
            dataset, loaded_labels, sizes, self.dictionary,
        )

    def extra_meters(self):
        return {
            'mcc': MCCMeter(),
            'acc': AccuracyMeter()
        }

    def aggregate_extra_metrics(self, logs):
        return {
            'mcc': tuple(
                reduce(lambda q, w: (sum(x) for x in zip(q, w)), [log['extra_metrics']['mcc'] for log in logs])),
            'acc': tuple(
                reduce(lambda q, w: (sum(x) for x in zip(q, w)), [log['extra_metrics']['acc'] for log in logs]))
        }

    def get_loss(self, model, criterion, sample, is_valid=False):
        loss, sample_size, logging_output = criterion(model, sample, reduce=not is_valid)

        if is_valid:
            probs = (-loss).exp()
            pos = sample['target'].view(-1).eq(1)
            neg = sample['target'].view(-1).eq(0)
            tp = (probs[pos] > 1 / self.num_labels).long().sum()
            tn = (probs[neg] > 1 / self.num_labels).long().sum()
            fp = neg.long().sum() - tn
            fn = pos.long().sum() - tp

            logging_output['extra_metrics'] = {
                'mcc': (tp.item(), tn.item(), fp.item(), fn.item()),
                'acc': (tp.item(), tn.item(), fp.item(), fn.item()),
            }

            loss = loss.sum()
            logging_output['loss'] = loss.item()

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
