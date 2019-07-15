#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os

from fairseq.tasks import FairseqTask, register_task
from fairseq.data import Dictionary, iterators
from fairseq.data.fb_conversations.fb_conversation_dataset import ConversationDataset
from fairseq.data.fb_conversations.fb_special_symbols import SpecialConversationSymbols
from fairseq.data.fb_hive_dataset import HiveDataset

logger = logging.getLogger(__name__)

MINIMUM_MESSAGE_COUNT = 4
PARTITION_DS = '2019-06-30'
SAMPLE_SIZE_LIMIT = 500000
SPLIT_PROPORTIONS = {
    'train': 0.8,
    'valid': 0.1,
    'test': 0.1,
}


@register_task('fb_conversation_modeling')
class FBConversationModelingTask(FairseqTask):
    """
    Train a language model from conversations using Facebook's infrastructure.

    Args:
        full_dataset (~fairseq.data.ConversationDataset): the data for the input
            of the language model, from which the train/valid/test will be taken

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        # fmt: on

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        assert args.data is not None
        symbols = [
            SpecialConversationSymbols.BOC,
            SpecialConversationSymbols.EOC,
            SpecialConversationSymbols.BOS0,
            SpecialConversationSymbols.EOS0,
            SpecialConversationSymbols.BOS1,
            SpecialConversationSymbols.EOS1,
        ]
        dictionary = Dictionary(extra_special_symbols=symbols)
        dictionary.add_from_file(
            f=os.path.join(args.data, 'dict.txt'),
        )
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # Divide data based on split

        split_portion = SPLIT_PROPORTIONS[split] * 10
        start = end = 0
        if split == 'train':
            start = 0
            end = split_portion - 1
        elif split == 'valid':
            # start right after train
            start = SPLIT_PROPORTIONS['train'] * 10
            end = start + (split_portion - 1)
        elif split == 'test':
            end = 9
            start = end - (split_portion - 1)
        else:
            logger.error("Invalid dataset split: {split}".format(**locals()))

        hive = HiveDataset(
            table='fair_conversation_samples',
            namespace='messages',
            limit=SAMPLE_SIZE_LIMIT,
            where_clause="ds = '{}' AND message_count > {}".format(
                PARTITION_DS,
                MINIMUM_MESSAGE_COUNT
            ),
        )
        conversation = ConversationDataset(
            dataset=hive,
            dictionary=self.dictionary,
            split_range=(start, end),
        )

        self.datasets[split] = conversation
        print('split data: {split}, {start}, {end}'.format(**locals()))

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        return iterators.StreamingEpochBatchIterator(
            dataset=dataset,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
        )
