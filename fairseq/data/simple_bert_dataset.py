# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import numpy as np
import torch

from . import data_utils, FairseqDataset


class BlockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokens,
        sizes,
        block_size,
        pad,
        cls,
        mask,
        sep,
        break_mode="doc"
    ):
        super().__init__()
        self.tokens = tokens
        self.total_size = len(tokens)
        self.pad = pad
        self.cls = cls
        self.mask = mask
        self.sep = sep
        self.block_indices = []
        self.break_mode = break_mode

        assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
        max_num_tokens = block_size - 2
        self.sents = []
        self.sizes = []

        if break_mode == "sent":
            curr = 0
            for sz in sizes:
                if sz == 0:
                    continue
                self.block_indices.append((curr, curr + sz))
                curr += sz
            for curr in range(len(self.block_indices)):
                sent = self.block_indices[curr]
                if sent[1] - sent[0] <= max_num_tokens:
                    self.sents.append(sent)
                    self.sizes.append(sent[1] - sent[0] + 2)
                    if len(self.sents) <= 5 or self.sizes[-1] > 512:
                        print("Sentence: %s (sz = %d)" % (self.sents[-1], self.sizes[-1]))
        elif break_mode == "doc":
            curr = 0
            cur_doc = []
            for sz in sizes:
                if sz == 0:
                    if len(cur_doc) == 0: continue
                    self.block_indices.append(cur_doc)
                    cur_doc = []
                else:
                    cur_doc.append((curr, curr + sz))
                curr += sz
            for doc in self.block_indices:
                current_chunk = []
                curr = 0
                while curr < len(doc):
                    sent = doc[curr]
                    if sent[1] - sent[0] <= max_num_tokens:
                        current_chunk.append(sent)
                        current_length = current_chunk[-1][1] - current_chunk[0][0]
                        if curr == len(doc) - 1 or current_length > max_num_tokens:
                            if current_length > max_num_tokens:
                                current_chunk = current_chunk[:-1]
                                curr -= 1
                            if len(current_chunk) > 0:
                                sent = (current_chunk[0][0], current_chunk[-1][1])
                                self.sents.append(sent)
                                self.sizes.append(sent[1] - sent[0] + 2)
                                if len(self.sents) <= 5 or self.sizes[-1] > 512:
                                    print("Sentence: %s (sz = %d)" % (self.sents[-1], self.sizes[-1]))
                            current_chunk = []
                    curr += 1

        else:
            raise ValueError("break_mode = %s not supported." % self.break_mode)

    def __getitem__(self, index):
        return self.sents[index]

    def __len__(self):
        return len(self.sizes)


class SimpleBertDataset(FairseqDataset):
    """
    A wrapper around BlockDataset for BERT data.
    Args:
        dataset (BlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, sizes, vocab, shuffle, seed, fix_seed, mask_ratio=0.15):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.mask_ratio = mask_ratio
        self.fix_seed = fix_seed

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            sent = self.dataset[index]

        masked_sent, masked_tgt = \
            self._mask_block(self.dataset.tokens[sent[0]:sent[1]], index, mask_ratio=self.mask_ratio)

        item = np.concatenate(
            [
                [self.vocab.cls()],
                masked_sent,
                [self.vocab.sep()],
            ]
        )
        target = np.concatenate([[self.vocab.pad()], masked_tgt, [self.vocab.pad()]])
        seg = np.zeros(sent[1] - sent[0] + 2)
        return {
            'id': index,
            'source': torch.from_numpy(item).long(),
            'segment_labels': torch.from_numpy(seg).long(),
            'lm_target': torch.from_numpy(target).long()
        }

    def __len__(self):
        return len(self.dataset)

    def _collate(self, samples, pad_idx):
        if len(samples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'ntokens': sum(len(s['source']) for s in samples),
            'net_input': {
                'src_tokens': merge('source'),
                'segment_labels': merge('segment_labels'),
            },
            'lm_target': merge('lm_target'),
            'nsentences': samples[0]['source'].size(0),
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return self._collate(samples, self.vocab.pad())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=12):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        source = self.vocab.dummy_sentence(tgt_len)
        segment_labels = torch.zeros(tgt_len, dtype=torch.long)
        lm_target = source
        bsz = num_tokens // tgt_len

        return self.collater([
            {
                'id': i,
                'source': source,
                'segment_labels': segment_labels,
                'lm_target': lm_target
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            return np.random.permutation(len(self))
        order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def _mask_block(self, sentence, idx, mask_ratio=.15):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        sentence = np.copy(sentence)
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * mask_ratio)
        if self.fix_seed:
            random = np.random.RandomState(2+idx)
            mask = random.choice(sent_length, mask_num, replace=False)
        else:
            mask = np.random.choice(sent_length, mask_num, replace=False)
        target = np.copy(sentence)
        for i in range(sent_length):
            if i in mask:
                if self.fix_seed:
                    rand = random.random_sample()
                else:
                    rand = np.random.random_sample()
                if rand < 0.8:
                    sentence[i] = self.vocab.mask()
                elif rand < 0.9:
                    # sample random token according to input distribution
                    sentence[i] = np.random.choice(self.dataset.tokens)
            else:
                target[i] = self.vocab.pad()
        return sentence, target
