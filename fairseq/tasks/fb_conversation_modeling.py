#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np
import torch.utils.data
from fairseq.data import Dictionary, ListDataset, iterators
from fairseq.data.fb_conversations.fb_conversation_dataset import ConversationDataset
from fairseq.data.fb_conversations.fb_special_symbols import SpecialConversationSymbols
from fairseq.data.fb_hive_dataset import HiveDataset, StreamingHiveDataset
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


def _date_list_from_arg(date_ranges: str) -> List[Tuple[str, str]]:
    return [tuple(r.split("|")) for r in date_ranges.split(",")]


def _shuffle(l: List) -> List:
    shuffled_indices = np.random.choice(
        list(range(len(l))),
        len(l),
        replace=False,
    )
    return [l[i] for i in shuffled_indices]


@register_task("fb_conversation_modeling")
class BaseConversationTask(LegacyFairseqTask):
    """
    Train a language model from conversations using Facebook's infrastructure.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('dictionary', help='path to dictionary directory')
        parser.add_argument('--table', type=str,
                            help='Hive table containing source data.')
        parser.add_argument('--namespace', type=str,
                            help='Hive namespace of table.')
        parser.add_argument('--query-limit', default=1500000, type=int,
                            help='Max samples to query per ds partition.')
        parser.add_argument('--train-date-range', type=str,
                            help='First and last partition ds to read from during training, '
                                 'in yyyy-mm-dd|yyyy-mm-dd format. e.g. 2019-01-01|2019-01-31')
        parser.add_argument('--eval-data-strategy', default='dates', choices=['dates', 'even-slice'],
                            help='dates requires specific date ranges from which to take the valid'
                                 'and test data. even-slices will query across all available dates until '
                                 'query-limit is hit.')
        parser.add_argument('--eval-date-ranges', type=str,
                            help='Partition ds ranges to read from during validation or test. Dates '
                                 'in range are separated with |, and ranges are separated with '
                                 'commas. e.g. 2019-01-13|2019-01-20,2019-01-25|2019-01-31')
        parser.add_argument('--data-loading', default='stream',
                            choices=['stream', 'preload'],
                            help='Preload data into memory all at once or stream')
        parser.add_argument('--new-data-date-range', type=str,
                            help='Date range that represents training data that hasn\'t been seen before.')
        parser.add_argument('--old-to-new-ratio', type=float, default=6,
                            help='Ratio of data to be trained on from new dates vs old dates')
        parser.add_argument('--split-pcts', type=str, default='0.998,0.001,0.001',
                            help='Comma separated list of ratios of data to be taken '
                                 'from train, valid, and test respectively. e.g. 0.998,0.001,0.001')
        # fmt: on

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        assert args.dictionary is not None
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
            f=os.path.join(args.dictionary, "dict.txt"),
        )
        logger.info("dictionary: {} types".format(len(dictionary)))

        if args.data_loading == "stream":
            return StreamingConversationTask(args, dictionary)
        elif args.data_loading == "preload":
            return PreloadConversationTask(args, dictionary)
        else:
            raise Exception(f"Unsupported data loading strategy: {args.data_loading}")

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.args = args
        assert sum(float(x) for x in args.split_pcts.split(",")) == 1.0

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
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
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, ConversationDataset)
        return iterators.StreamingEpochBatchIterator(
            dataset=dataset,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
        )

    def _prob_range_from_split(self, split) -> Tuple[float, float]:
        trn_pct, vld_pct, tst_pct = [float(x) for x in self.args.split_pcts.split(",")]

        # (start, end]
        start = end = 0.0
        if split == "train":
            end = trn_pct
        elif split == "valid":
            # start right after train
            start = trn_pct
            end = start + vld_pct
        elif split == "test":
            start = trn_pct + vld_pct
            end = 1.0
        else:
            logger.error(f"Invalid dataset split: {split}")

        logger.info(f"Probability range for split: {split}, {start}, {end}")

        return start, end


def _should_include(key, split_range) -> bool:
    random_id = abs(hash(key))
    return split_range[0] <= (random_id % 10) <= split_range[1]


class PreloadConversationTask(BaseConversationTask):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def _set_up_train_dataset(self, split_range) -> torch.utils.data.Dataset:
        new_date_ranges = _date_list_from_arg(self.args.new_data_date_range)
        logger.info(f"Setting up training data: {split_range}, {new_date_ranges}")
        new_hive_data = HiveDataset(
            table=self.args.table,
            namespace=self.args.namespace,
            limit=self.args.query_limit,
            date_ranges=new_date_ranges,
            filter_fn=lambda x: _should_include(x[0], split_range),
        )

        desired_total_data_size = self.args.old_to_new_ratio * len(new_hive_data)
        desired_old_data_size = (
            1 - (1 / self.args.old_to_new_ratio)
        ) * desired_total_data_size

        old_date_ranges = _date_list_from_arg(self.args.train_date_range)
        old_hive_data = HiveDataset(
            table=self.args.table,
            namespace=self.args.namespace,
            limit=min(self.args.query_limit, desired_old_data_size),
            date_ranges=old_date_ranges,
            filter_fn=lambda x: _should_include(x[0], split_range),
        )

        old_hive_data = old_hive_data[: int(desired_old_data_size)]

        all_data = new_hive_data.data + list(old_hive_data)
        conversations = ConversationDataset(
            dataset=ListDataset(dataset=_shuffle(all_data)),
            dictionary=self.dictionary,
            split_range=split_range,
        )
        logger.info(
            f"Created train dataset of size: {len(conversations)} conversations"
        )

        return conversations

    def load_dataset(self, split, combine=False, **kwargs):
        prob_range = self._prob_range_from_split(split)
        date_ranges = []
        if split == "train":
            self.datasets[split] = self._set_up_train_dataset(prob_range)
            return
        elif split == "valid" or split == "test":
            date_ranges = _date_list_from_arg(self.args.eval_date_ranges)
        else:
            logger.error("Invalid dataset split: {split}".format(**locals()))

        logger.info(
            f"Data split: {split}, {prob_range[0]}, {prob_range[1]}, {date_ranges}"
        )

        hive_dataset = HiveDataset(
            table=self.args.table,
            namespace=self.args.namespace,
            limit=self.args.query_limit,
            date_ranges=date_ranges,
        )
        conversation = ConversationDataset(
            dataset=hive_dataset,
            dictionary=self.dictionary,
            split_range=prob_range,
        )
        self.datasets[split] = conversation

    def dataset(self, split):
        return self.datasets[split]


class StreamingConversationTask(BaseConversationTask):
    def load_dataset(self, split, combine=False, **kwargs):
        # Dataset does not need to be loaded since the data is streamed.
        # This task overrides dataset() instead.
        pass

    def dataset(self, split):
        return self._train_dataset() if split == "train" else self._eval_dataset(split)

    def _train_dataset(self):
        start, end = self._prob_range_from_split("train")
        date_ranges = self._date_range_for_split("train")
        fresh_ranges = None
        if self.args.new_data_date_range:
            fresh_ranges = _date_list_from_arg(self.args.new_data_date_range)

        hive = StreamingHiveDataset(
            table=self.args.table,
            namespace=self.args.namespace,
            limit=self.args.query_limit,
            date_ranges=date_ranges,
            fresh_date_ranges=fresh_ranges,
            fresh_ratio=self.args.old_to_new_ratio,
            shuffle=True,
        )
        conversation = ConversationDataset(
            dataset=hive,
            dictionary=self.dictionary,
            split_range=(start, end),
        )
        return conversation

    def _eval_dataset(self, split):
        start, end = self._prob_range_from_split(split)
        date_ranges = self._date_range_for_split(split)

        hive = StreamingHiveDataset(
            table=self.args.table,
            namespace=self.args.namespace,
            limit=self.args.query_limit,
            date_ranges=date_ranges,
        )
        conversation = ConversationDataset(
            dataset=hive,
            dictionary=self.dictionary,
            split_range=(start, end),
        )
        return conversation

    def _date_range_for_split(self, split) -> Optional[List[Tuple[str, str]]]:
        is_eval = split == "valid" or split == "test"
        if is_eval and self.args.eval_data_strategy != "dates":
            return None

        date_range = None
        if split == "train":
            date_range = self.args.train_date_range
        elif is_eval:
            date_range = self.args.eval_date_ranges
        else:
            logger.error(f"Invalid dataset split: {split}")

        return _date_list_from_arg(date_range) if date_range else None
