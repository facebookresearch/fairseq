# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq.data import FairseqDataset


logger = logging.getLogger(__name__)


def collate(
    samples,
    tokenizer
):
    if len(samples) == 0:
        return {}
    #logger.info("==" * 100 + "{}".format(os.getpid()))

    """
    sample = {
        'id': index,
        'source': src_item, # str
        'source_lengths': src_size,  # np
        'target': tgt_item, # np
    }
    """

    def merge_src_str(key):
        """
        return examples like:
        {'input_ids': tensor([[101, 2644, 1962, 102, 0],
                              [101, 2769, 738, 1962, 102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0],
                                                                                       [0, 0, 0, 0, 0]]),
         'attention_mask': tensor([[1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1]])}
        """

        ret = tokenizer([s[key] for s in samples], padding=True, return_tensors='pt')
        return ret

    def merge_tgt_label(key):
        return torch.LongTensor([s[key] for s in samples])

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokened = merge_src_str('source')
    src_lengths = src_tokened['input_ids'].ne(0).sum(dim=1)

    target = None
    if samples[0].get('target', None) is not None:
        target = merge_tgt_label('target')
    ntokens = src_lengths.sum().item()

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'input_ids': src_tokened['input_ids'],
            'token_type_ids': src_tokened['token_type_ids'],
            'attention_mask': src_tokened['attention_mask'],
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    return batch


class MarianTokenizerDataset(FairseqDataset):
    """
        Simple dataset for demo
    """
    def __init__(self, src, tgt=None, tokenizer=None, shuffle=True):
        if tgt is not None:
            assert len(src) == len(tgt), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.src_sizes = np.array(self.src.sizes) + 2
        self.shuffle = shuffle

    #def get_batch_shapes(self):
    #    raise NotImplementedError()

    # def batch_by_size(
    #     self,
    #     indices,
    #     max_tokens=None,
    #     max_sentences=None,
    #     required_batch_size_multiple=1,
    # ):
    #     logger.info("indices:{} max_tokens:{} max_sentences:{} required_batch_size_multiple:{}".format(indices, max_tokens, max_sentences, required_batch_size_multiple))
    #     raise NotImplementedError()

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        src_size = self.src_sizes[index]

        # wrap single raw data
        example = {
            'id': index,
            'source': src_item, # str
            'source_lengths': src_size,  # np
            'target': tgt_item, # np
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        """
        res = collate(
            samples,
            self.tokenizer
        )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # raise NotImplementedError
        return self.src_sizes[index]
        #return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        #return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        # raise NotImplementedError("check this")
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices

    @property
    def supports_prefetch(self):
        return False

    def filter_indices_by_size(self, indices, max_sizes):
        """ Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if max_sizes is None:
            return indices, []

        if type(max_sizes) in (int, float):
            max_src_size = max_sizes
        else:
            max_src_size = max_sizes[0]

        ignored = indices[self.src_sizes[indices] > max_src_size]

        if len(ignored) > 0:
            indices = indices[self.src_sizes[indices] <= max_src_size]
        return indices, ignored.tolist()
