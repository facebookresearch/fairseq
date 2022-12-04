#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from fairseq.data import FairseqDataset, FairseqIterableDataset, data_utils, encoders
from fairseq.data.fb_conversations.fb_special_symbols import SpecialConversationSymbols
from fairseq.data.fb_hive_dataset import HiveDataset

logger = logging.getLogger("fairseq.fb_conversation_dataset")


def _should_include(key: str, split_range: Tuple[float, float]) -> bool:
    """
    Hashes key to decimal between 0 and 1 and returns whether it falls
    within the supplied range.
    """
    max_precision_order = 10000
    decimal_hash = (hash(key) % max_precision_order) / max_precision_order
    return split_range[0] < decimal_hash <= split_range[1]


def _tokenize_and_reformat_conversations(item, dictionary, encoder) -> Dict[str, Any]:
    """
    Given an input (*item*) of the form:
        (
            '123:124', <- thread key
            [[
                1558359573 <- timestamp
                1, <- user ID
                'hello there', <- message body
            ], ...],
            3, <- message count
            '2019-06-30' <- partition
        )
    this will return:
        {
            'id': 123124,
            'source': tensor([4, 6, 31373, 612, 7, ..., 5])
        }
    """
    if item is None:
        return item

    # Verify data is in expected format
    assert len(item) == 4
    assert (
        isinstance(item[0], str)
        and isinstance(item[1], list)
        and isinstance(item[2], int)
    )

    def _reformat_msg(msg, sender_list, encoder):
        sender = msg[1]

        body = encoder.encode(msg[2])
        body = dictionary.encode_line(
            body,
            add_if_not_exist=False,
            append_eos=False,
        )

        # Add sender to shared list if not there already
        if sender not in sender_list:
            sender_list.append(sender)

        # Make new sender ID based on index in list, so first person to
        # talk will be s0
        user_id_short = str(sender_list.index(sender))

        bos = dictionary.index("<s{user_id_short}>".format(**locals()))
        eos = dictionary.index("</s{user_id_short}>".format(**locals()))

        return torch.cat([torch.IntTensor([bos]), body, torch.IntTensor([eos])])

    try:
        # Convert text thread key into an int ('1:2' -> 12)
        id = int("".join(item[0].split(":")))
        # Join all the messages into a single tensor separated by sender tags
        user_list = []
        convo_tensor = torch.cat(
            [torch.IntTensor(_reformat_msg(m, user_list, encoder)) for m in item[1]]
        )
        # Create final tensor by bookending conversation tags
        boc = dictionary.index(SpecialConversationSymbols.BOC)
        eoc = dictionary.index(SpecialConversationSymbols.EOC)
        item = {
            "id": id,
            "source": torch.cat(
                [torch.IntTensor([boc]), convo_tensor, torch.IntTensor([eoc])]
            ),
        }
    except Exception as e:
        logger.error("Exception: {}\n{}".format(e, traceback.format_exc()))
        return None
    return item


def _torchify(item, dictionary) -> Dict[str, Any]:
    """
    Converts item into a format usable by PyTorch.

    Given an (*item*) of the form:
        {
            'id': 123124,
            'source': tensor([4, 6, 31373, 612, 7, ..., 5])
        }
    this will return:
        {
            'id': tensor([123124]),
            'ntokens': 37,
            'net_input': {
                'src_tokens': tensor([5, 4, 6, 31373, 612, ..., 53])
            },
            'target': tensor([4, 6, 31373, 612, 7, ..., 5])
        }
    """
    tokenized_conversation = item["source"].long()
    ntokens = len(tokenized_conversation)
    if ntokens > 1024 or ntokens < 20:
        logger.info("Skipped conversation with token length: {}".format(ntokens))
        return None
    source = data_utils.collate_tokens(
        [tokenized_conversation],
        dictionary.pad(),
        eos_idx=dictionary.index(SpecialConversationSymbols.EOC),
        move_eos_to_beginning=True,
    )
    target = data_utils.collate_tokens(
        [tokenized_conversation],
        dictionary.pad(),
    )
    torch_item = {
        # Bound ID to 64 bit max to avoid overflow
        "id": torch.LongTensor([item["id"] % (2**63 - 1)]),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": source,
            "src_lengths": torch.LongTensor([ntokens]),
        },
        "target": target,
    }
    return torch_item


class ConversationDataset(FairseqDataset, FairseqIterableDataset):
    """
    A dataset representing conversations between two or more people.

    Given a dataset with items of the form:
        (
            '123:124', <- thread key
            [[
                1558359573 <- timestamp
                1, <- user ID
                'hello there', <- message body
            ], ...],
            3, <- message count
            '2019-06-30' <- partition
        )
    this will items like:
        {
            'id': tensor([123124]),
            'ntokens': 37,
            'net_input': {
                'src_tokens': tensor([5, 4, 6, 31373, 612, ..., 53])
            },
            'target': tensor([4, 6, 31373, 612, 7, ..., 5])
        }

    Args:
        dataset (torch.utils.data.Dataset): dataset to reformat
        dictionary (fairseq.data.Dictionary): pre-made dictionary for the task
        split_range (tuple(int, int)): Inclusive range between 0 and 9 from
            which to sample. (e.g. (0, 7) will sample 80% of the data)
    """

    def __init__(
        self,
        dataset: HiveDataset,
        dictionary,
        split_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.dataset = dataset
        self.dictionary = dictionary
        self.split_range = split_range
        from fairseq.data.encoders.gpt2_bpe import GPT2BPEConfig

        bpe_cfg = GPT2BPEConfig(
            gpt2_encoder_json="/mnt/vol/gfsai-flash3-east/ai-group/users/myleott/gpt2_bpe/encoder.json",
            gpt2_vocab_bpe="/mnt/vol/gfsai-flash3-east/ai-group/users/myleott/gpt2_bpe/vocab.bpe",
        )
        bpe_cfg._name = "gpt2"
        self.bpe = encoders.build_bpe(bpe_cfg)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self._transform_item(self.dataset[index])
        elif isinstance(index, slice):
            return ConversationDataset(self.dataset[index], self.dictionary)
        else:
            raise TypeError(
                "Index must be int or slice, not {}".format(type(index).__name__)
            )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        item = self[index]
        if item is None:
            return 0
        return item["ntokens"]

    def __len__(self):
        # We'll only look at a subset of the dataset as determined by the split
        # range, and we should reflect that in the length.
        ratio_of_data = self.split_range[1] - self.split_range[0]
        return int(len(self.dataset) * ratio_of_data)

    def __iter__(self):
        for x in self.dataset:
            if not _should_include(x[0], self.split_range):
                continue
            item = self._transform_item(x)
            if item is not None:
                yield item

    def _transform_item(self, item):
        return _torchify(
            _tokenize_and_reformat_conversations(
                item,
                self.dictionary,
                self.bpe,
            ),
            self.dictionary,
        )
