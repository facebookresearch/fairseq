# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq.data import Dictionary, FairseqDataset
from fairseq.tasks import FairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task('dummy_mt')
class DummyMTTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--dict-size', default=49996, type=int)
        parser.add_argument('--dataset-size', default=100000, type=int)
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        dictionary.pad_to_multiple_(8)  # often faster if divisible by 8

        seq = torch.arange(args.tokens_per_sample + 1) + dictionary.pad() + 1

        self.dummy_src = seq[:-1]
        self.dummy_tgt = seq[1:]

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task. """
        dictionary = Dictionary()
        for i in range(args.dict_size):
            dictionary.add_symbol('word{}'.format(i))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.args.max_sentences is not None:
            bsz = self.args.max_sentences
        else:
            bsz = max(1, self.args.max_tokens // self.args.tokens_per_sample)
        tgt = torch.stack([self.dummy_tgt for _ in range(bsz)])
        self.datasets[split] = DummyDataset(
            {
                'id': 1,
                'net_input': {
                    'src_tokens': torch.stack([self.dummy_src for _ in range(bsz)]),
                    'src_lengths': torch.full(
                        (bsz, ), self.args.tokens_per_sample, dtype=torch.long
                    ),
                    'prev_output_tokens': tgt.clone(),
                },
                'target': tgt,
                'nsentences': bsz,
                'ntokens': bsz * self.args.tokens_per_sample,
            },
            num_items=self.args.dataset_size,
            item_size=self.args.tokens_per_sample,
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class DummyDataset(FairseqDataset):

    def __init__(self, batch, num_items, item_size):
        super().__init__()
        self.batch = batch
        self.num_items = num_items
        self.item_size = item_size

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.num_items

    def collater(self, samples):
        return self.batch

    @property
    def sizes(self):
        return np.array([self.item_size] * self.num_items)

    def num_tokens(self, index):
        return self.item_size

    def size(self, index):
        return self.item_size

    def ordered_indices(self):
        return np.arange(self.num_items)

    @property
    def supports_prefetch(self):
        return False
