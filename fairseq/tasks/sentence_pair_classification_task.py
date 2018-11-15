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
    SentencePairClassificationDataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import ClassificationMeter, RegressionMeter

from . import FairseqTask, register_task


@register_task('sentence_pair_classification')
class SentencePairClassificationTask(FairseqTask):
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
        parser.add_argument('--num-labels', type=int, default=3,
                            help='number of labels')
        parser.add_argument('--concat-sentences-mode', default='none',
                            help='concat sentences in the dataset. none = dont concat, eos = eos concat, unk = unk concat')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.padding_idx = -100
        self.num_labels = args.num_labels
        self.concat_sentences_mode = args.concat_sentences_mode

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
            path1 = os.path.join(base_path + '_s1')
            path2 = os.path.join(base_path + '_s2')

            for path, datasets in zip([path1, path2], loaded_datasets):
                if self.args.raw_text and IndexedRawTextDataset.exists(path):
                    ds = IndexedRawTextDataset(path, self.dictionary)
                elif not self.args.raw_text and IndexedInMemoryDataset.exists(path):
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
                cast = int if self.num_labels > 0 else float
                loaded_labels.extend(cast(l) for l in lines)

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[0][-1])))

            if not combine:
                break

        if self.num_labels == 2:
            loaded_labels = [l if l == 1 else 0 for l in loaded_labels]

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

        self.datasets[split] = SentencePairClassificationDataset(
            dataset1, dataset2, loaded_labels, sizes1, sizes2, self.dictionary, self.concat_sentences_mode
        )

    def extra_meters(self):
        if self.num_labels > 0:
            return {
                'classification': ClassificationMeter(),
            }
        else:
            return {
                'regression': RegressionMeter()
            }

    def aggregate_extra_metrics(self, logs):
        if self.num_labels > 0:
            return {
                'classification': tuple(
                    reduce(lambda q, w: (sum(x) for x in zip(q, w)),
                           [log['extra_metrics']['classification'] for log in logs if 'extra_metrics' in log])),
                'misclassified': sum([log['extra_metrics']['misclassified'] for log in logs if 'extra_metrics' in log],
                                     [])
            }
        else:
            return {
                'regression': tuple(
                    reduce(lambda q, w: (sum(x, []) for x in zip(q, w)),
                           [log['extra_metrics']['regression'] for log in logs if 'extra_metrics' in log]))
            }

    def get_loss(self, model, criterion, sample, is_valid=False):
        loss, sample_size, logging_output = criterion(model, sample, reduce=not is_valid)

        if is_valid:
            if self.num_labels > 0:
                probs = (-loss).exp()

                tp = tn = fp = fn = 0

                correct_by_lbl = []

                if self.num_labels == 2:
                    pos = sample['target'].view(-1).eq(1)
                    neg = sample['target'].view(-1).eq(0)
                    correct_pos = probs[pos] > 1 / self.num_labels
                    correct_neg = probs[neg] > 1 / self.num_labels

                    correct_by_lbl.extend([(pos, correct_pos), (neg, correct_neg)])

                    tp = correct_pos.long().sum().item()
                    tn = correct_neg.long().sum().item()
                    fp = neg.long().sum().item() - tn
                    fn = pos.long().sum().item() - tp
                else:
                    for i in range(self.num_labels + 1):
                        pos = sample['target'].view(-1).eq(i)
                        correct_lbl = probs[pos] > 1 / self.num_labels

                        correct_by_lbl.append((pos, correct_lbl))

                        c_tp = correct_lbl.long().sum().item()
                        tp += c_tp
                        fn += pos.long().sum().item() - c_tp

                logging_output['extra_metrics'] = {
                    'classification': (tp, tn, fp, fn),
                    'misclassified': []
                }

                if False:
                    correct = correct_by_lbl[0][1].new_zeros(probs.shape)

                    for mask, corr in correct_by_lbl:
                        correct[mask] = corr

                    incorrect = ~correct
                    incorrect_ids = sample['id'][incorrect.nonzero()]
                    incorrect_probs = probs[incorrect.nonzero()]

                    logging_output['extra_metrics']['misclassified'] = list(
                        zip(incorrect_ids.squeeze(-1).tolist(),
                            map(lambda x: round(x, 2), incorrect_probs.squeeze(-1).tolist())))

            else:
                xs = logging_output['preds'].view(-1).tolist()
                ys = sample['target'].view(-1).tolist()

                logging_output['extra_metrics'] = {
                    'regression': (xs, ys),
                }

            loss = loss.sum()
            logging_output['loss'] = loss.item()

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
