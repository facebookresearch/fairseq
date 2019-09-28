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
import torch

from torch.utils.data import ConcatDataset

from fairseq.data import (
    Dictionary, IndexedInMemoryDataset,
    SquadDataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import ClassificationMeter

from . import FairseqTask, register_task

class BertDictionary(Dictionary):
    """Dictionary for BERT tasks
        extended from Dictionary by adding support for cls as well as mask symbols"""
    def __init__(
        self,
        pad='[PAD]',
        unk='[UNK]',
        cls='[CLS]',
        mask='[MASK]',
        sep='[SEP]'
    ):
        super().__init__(pad, unk)
        (
            self.cls_word,
            self.mask_word,
            self.sep_word
        ) = cls, mask, sep
        self.nspecial = len(self.symbols)

    def cls(self):
        """Helper to get index of cls symbol"""
        idx = self.add_symbol(self.cls_word)
        return idx

    def mask(self):
        """Helper to get index of mask symbol"""
        idx = self.add_symbol(self.mask_word)
        return idx

    def sep(self):
        """Helper to get index of sep symbol"""
        idx = self.add_symbol('[SEP]')
        return idx

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
        parser.add_argument('--data-file', type=str, help='the json data file to score (assumed to be in data dir)')
        parser.add_argument('--n-best-size', type=int, default=20)
        parser.add_argument('--gen-subset', type=str, default='valid')
        parser.add_argument('--max-length', type=int, default=384)
        parser.add_argument('--stride', type=int, default=128)
        parser.add_argument('--max-query-length', type=int, default=64)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.valid_groups = ('classification_start', 'classification_end')

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = [[], []]
        loaded_labels = []
        loaded_ids = []
        loaded_raw_actual_text = []
        loaded_tok_to_orig_index = []
        stop = False

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            base_path = os.path.join(self.args.data, split_k)
            path1 = os.path.join(base_path + '_1')
            path2 = os.path.join(base_path + '_2')

            for path, datasets in zip([path1, path2], loaded_datasets):
                if IndexedInMemoryDataset.exists(path):
                    ds = IndexedDataset(path, fix_lua_indexing=False)
                else:
                    if k > 0:
                        stop = True
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({}) {}'.format(split, self.args.data, path))

                datasets.append(
                    TokenBlockDataset(
                        ds, 0, pad=self.dictionary.pad(),
                        break_mode='eos', include_targets=False,
                    ))

            if stop:
                break
            with open(base_path + '.lbl', 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbls = [int(x) for x in line.strip().split()]
                    impossible = lbls[0] == 0
                    answers = [] if impossible else [lbls[1], lbls[2]]
                    
                    loaded_labels.append(answers)

            with open(base_path + '_3.txt', 'r') as act_f:
                lines = act_f.readlines()
                for line in lines:
                    loaded_raw_actual_text.append(line.strip())
            with open(base_path + '_4.txt', 'r') as idx_map_f:
                lines = idx_map_f.readlines()
                for line in lines:
                    idx_map = [int(x) for x in line.strip().split()]
                    loaded_tok_to_orig_index.append(idx_map)

            with open(base_path + '.id', 'r') as id_f:
                loaded_ids.extend([id.strip() for id in id_f.readlines()])

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

        assert len(dataset1) == len(loaded_labels)
        assert len(dataset2) == len(loaded_labels)

        self.datasets[split] = SquadDataset(
            dataset1, dataset2, loaded_labels, 
            loaded_ids, loaded_raw_actual_text, 
            loaded_tok_to_orig_index, sizes1, 
            sizes2, self.dictionary, self.args.stride,
            self.args.max_length, self.args.max_query_length
        )

    def extra_meters(self):
        return {
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
        loss, sample_size, logging_output, outs = criterion(model, sample)

        if is_valid:
            logging_output['extra_metrics'] = {}
            for g, o, t in zip(self.valid_groups, outs, sample['target']):
                t = t.squeeze(-1)
                pred_t = torch.argmax(o, dim=-1)
                tp = t.eq(pred_t).long().sum().item()
                tn = 0
                fp = t.size(0) - tp
                fn = 0

                logging_output['extra_metrics'][g] = (tp, tn, fp, fn)

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
