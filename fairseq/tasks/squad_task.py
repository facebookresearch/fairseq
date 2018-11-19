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
    Dictionary, IndexedInMemoryDataset,
    SquadDataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import ClassificationMeter, RegressionMeter

from . import FairseqTask, register_task


@register_task('squad')
class SquadTask(FairseqTask):
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
        parser.add_argument('--concat-sentences-mode', default='unk_only',
                            help='concat sentences in the dataset. eos = eos concat, '
                                 'unk = unk concat (with eos), unk_only = concat with unk')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.padding_idx = -100
        self.concat_sentences_mode = args.concat_sentences_mode
        self.valid_groups = ('classification_imp', 'classification_start', 'classification_end')

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

        loaded_datasets = [[], []]
        loaded_labels = []
        stop = False

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            base_path = os.path.join(self.args.data, split_k)
            path1 = os.path.join(base_path + '_1')
            path2 = os.path.join(base_path + '_2')

            for path, datasets in zip([path1, path2], loaded_datasets):
                if IndexedInMemoryDataset.exists(path):
                    ds = IndexedDataset(path, fix_lua_indexing=True)
                else:
                    if k > 0:
                        stop = True
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

                datasets.append(
                    TokenBlockDataset(
                        ds, 0, pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                        break_mode='eos', include_targets=False,
                    ))

            if stop:
                break
            with open(base_path + '.lbl', 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbls = [int(x) for x in line.strip().split()]
                    impossible = lbls[0] == 1
                    answers = [] if impossible else list(zip(lbls[1::2], lbls[2::2]))

                    loaded_labels.append(answers)

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[0][-1])))

            if not combine:
                break

        if len(loaded_datasets[0]) == 1:
            dataset1 = loaded_datasets[0][0]
            dataset2 = loaded_datasets[1][0]
            sizes1 = dataset1.sizes
            sizes2 = dataset2.sizes
        else:
            dataset1 = ConcatDataset(loaded_datasets[0])
            dataset2 = ConcatDataset(loaded_datasets[1])
            sizes1 = np.concatenate([ds.sizes for ds in loaded_datasets[0]])
            sizes2 = np.concatenate([ds.sizes for ds in loaded_datasets[1]])

        self.datasets[split] = SquadDataset(
            dataset1, dataset2, loaded_labels, sizes1, sizes2, self.dictionary, self.padding_idx,
            self.concat_sentences_mode
        )

    def extra_meters(self):
        return {
            'classification_imp': ClassificationMeter('imp'),
            'classification_start': ClassificationMeter('start'),
            'classification_end': ClassificationMeter('end'),
        }

    def aggregate_extra_metrics(self, logs):
        agg = {}
        for m in self.valid_groups:
            agg[m] = tuple(
                reduce(lambda q, w: (sum(x) for x in zip(q, w)),
                       [log['extra_metrics'][m] for log in logs if 'extra_metrics' in log]))
        return agg

    def get_loss(self, model, criterion, sample, is_valid=False):
        loss, sample_size, logging_output = criterion(model, sample, reduce=not is_valid)

        if is_valid:
            logging_output['extra_metrics'] = {}
            for g, l, t in zip(self.valid_groups, loss, sample['target']):
                probs = (-l).exp()
                pos = t.view(-1).eq(1)
                neg = t.view(-1).eq(0)

                # tp = (probs[pos] > 1 / self.num_labels).long().sum() if pos.any() else probs.new_zeros(1).long()
                # tn = (probs[neg] > 1 / self.num_labels).long().sum() if neg.any() else probs.new_zeros(1).long()

                num_labels = t.size(1)

                correct_pos = probs[pos] > 0.5
                correct_neg = probs[neg] > 0.5

                tp = correct_pos.long().sum()
                tn = correct_neg.long().sum()

                fp = neg.long().sum() - tn
                fn = pos.long().sum() - tp

                logging_output['extra_metrics'][g] = (tp.item(), tn.item(), fp.item(), fn.item())

            loss = logging_output['loss']

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
