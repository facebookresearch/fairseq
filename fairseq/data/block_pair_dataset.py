# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import numpy as np
import torch

from . import FairseqDataset


class BlockPairDataset(FairseqDataset):
    """Break a Dataset of tokens into sentence pair blocks for next sentence
       prediction as well as masked language model.

       High-level logics are:
       1. break input tensor to tensor blocks
       2. pair the blocks with 50% next sentence and 50% random sentence
       3. return paired blocks as well as related segment labels

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes: array of sentence lengths
        dictionary: dictionary for the task
        block_size: maximum block size
        break_mode: mode for breaking copurs into block pairs. currently we support
            2 modes
            doc: respect document boundaries and each part of the pair should belong to on document
            none: don't respect any boundary and cut tokens evenly
        short_seq_prob: probability for generating shorter block pairs
    """

    def __init__(
        self,
        dataset,
        dictionary,
        sizes,
        block_size,
        break_mode="doc",
        short_seq_prob=0.1,
    ):
        super().__init__()
        self.dataset = dataset
        self.pad = dictionary.pad()
        self.eos = dictionary.eos()
        self.cls = dictionary.cls()
        self.mask = dictionary.mask()
        self.sep = dictionary.sep()
        self.break_mode = break_mode
        self.dictionary = dictionary
        self.short_seq_prob = short_seq_prob
        self.block_indices = []

        assert len(dataset) == len(sizes)

        if break_mode == "doc":
            cur_doc = []
            for sent_id, sz in enumerate(sizes):
                # empty line as document separator
                if sz == 0:
                    if len(cur_doc) == 0:
                        continue
                    self.block_indices.append(cur_doc)
                    cur_doc = []
                else:
                    cur_doc.append(sent_id)
            max_num_tokens = block_size - 3  # Account for [CLS], [SEP], [SEP]
            self.sent_pairs = []
            self.sizes = []
            for doc_id, doc in enumerate(self.block_indices):
                self._generate_sentence_pair(doc, doc_id, max_num_tokens, sizes)
        elif break_mode is None or break_mode == "none":
            # each block should have half of the block size since we are constructing block pair
            sent_length = (block_size - 3) // 2
            total_len = sum(dataset.sizes)
            length = math.ceil(total_len / sent_length)

            def block_at(i):
                start = i * sent_length
                end = min(start + sent_length, total_len)
                return (start, end)

            sent_indices = np.array([block_at(i) for i in range(length)])
            sent_sizes = np.array([e - s for s, e in sent_indices])
            dataset_index = self._sent_to_dataset_index(sent_sizes)

            # pair sentences
            self._pair_sentences(dataset_index)
        else:
            raise ValueError("Invalid break_mode: " + break_mode)

    def _pair_sentences(self, dataset_index):
        """
        Give a list of evenly cut blocks/sentences, pair these sentences with 50%
        consecutive sentences and 50% random sentences.
        This is used for none break mode
        """
        # pair sentences
        for sent_id, sent in enumerate(dataset_index):
            next_sent_label = (
                1 if np.random.rand() > 0.5 and sent_id != len(dataset_index) - 1 else 0
            )
            if next_sent_label:
                next_sent = dataset_index[sent_id + 1]
            else:
                next_sent = dataset_index[
                    self._skip_sampling(len(dataset_index), [sent_id, sent_id + 1])
                ]
            self.sent_pairs.append((sent, next_sent, next_sent_label))

            # The current blocks don't include the special tokens but the
            # sizes already account for this
            self.sizes.append(3 + sent[3] + next_sent[3])

    def _sent_to_dataset_index(self, sent_sizes):
        """
        Build index mapping block indices to the underlying dataset indices
        """
        dataset_index = []
        ds_idx, ds_remaining = -1, 0
        for to_consume in sent_sizes:
            sent_size = to_consume
            if ds_remaining == 0:
                ds_idx += 1
                ds_remaining = sent_sizes[ds_idx]
            start_ds_idx = ds_idx
            start_offset = sent_sizes[ds_idx] - ds_remaining
            while to_consume > ds_remaining:
                to_consume -= ds_remaining
                ds_idx += 1
                ds_remaining = sent_sizes[ds_idx]
            ds_remaining -= to_consume
            dataset_index.append(
                (
                    start_ds_idx,  # starting index in dataset
                    start_offset,  # starting offset within starting index
                    ds_idx,  # ending index in dataset
                    sent_size,  # sentence length
                )
            )
        assert ds_remaining == 0
        assert ds_idx == len(self.dataset) - 1
        return dataset_index

    def _generate_sentence_pair(self, doc, doc_id, max_num_tokens, sizes):
        """
        Go through a single document and genrate sentence paris from it
        """
        current_chunk = []
        current_length = 0
        curr = 0
        # To provide more randomness, we decrease target seq length for parts of
        # samples (10% by default). Note that max_num_tokens is the hard threshold
        # for batching and will never be changed.
        target_seq_length = max_num_tokens
        if np.random.random() < self.short_seq_prob:
            target_seq_length = np.random.randint(2, max_num_tokens)
        # loop through all sentences in document
        while curr < len(doc):
            sent_id = doc[curr]
            current_chunk.append(sent_id)
            current_length = sum(sizes[current_chunk])
            # split chunk and generate pair when exceed target_seq_length or
            # finish the loop
            if curr == len(doc) - 1 or current_length >= target_seq_length:
                # split the chunk into 2 parts
                a_end = 1
                if len(current_chunk) > 2:
                    a_end = np.random.randint(1, len(current_chunk) - 1)
                sent_a = current_chunk[:a_end]
                len_a = sum(sizes[sent_a])
                # generate next sentence label, note that if there is only 1 sentence
                # in current chunk, label is always 0
                next_sent_label = (
                    1 if np.random.rand() > 0.5 and len(current_chunk) != 1 else 0
                )
                if not next_sent_label:
                    # if next sentence label is 0, sample sent_b from a random doc
                    target_b_length = target_seq_length - len_a
                    rand_doc_id = self._skip_sampling(len(self.block_indices), [doc_id])
                    random_doc = self.block_indices[rand_doc_id]
                    random_start = np.random.randint(0, len(random_doc))
                    sent_b = []
                    len_b = 0
                    for j in range(random_start, len(random_doc)):
                        sent_b.append(random_doc[j])
                        len_b = sum(sizes[sent_b])
                        if len_b >= target_b_length:
                            break
                    # return the second part of the chunk since it's not used
                    num_unused_segments = len(current_chunk) - a_end
                    curr -= num_unused_segments
                else:
                    # if next sentence label is 1, use the second part of chunk as sent_B
                    sent_b = current_chunk[a_end:]
                    len_b = sum(sizes[sent_b])
                # currently sent_a and sent_B may be longer than max_num_tokens,
                # truncate them and return block idx and offsets for them
                sent_a, sent_b = self._truncate_sentences(
                    sent_a, sent_b, max_num_tokens
                )
                self.sent_pairs.append((sent_a, sent_b, next_sent_label))
                self.sizes.append(3 + sent_a[3] + sent_b[3])
                current_chunk = []
            curr += 1

    def _skip_sampling(self, total, skip_ids):
        """
        Generate a random integer which is not in skip_ids. Sample range is [0, total)
        TODO: ids in skip_ids should be consecutive, we can extend it to more generic version later
        """
        rand_id = np.random.randint(total - len(skip_ids))
        return rand_id if rand_id < min(skip_ids) else rand_id + len(skip_ids)

    def _truncate_sentences(self, sent_a, sent_b, max_num_tokens):
        """
        Trancate a pair of sentence to limit total length under max_num_tokens
        Logics:
            1. Truncate longer sentence
            2. Tokens to be truncated could be at the beginning or the end of the sentnce
        Returns:
            Truncated sentences represented by dataset idx
        """
        len_a, len_b = sum(self.dataset.sizes[sent_a]), sum(self.dataset.sizes[sent_b])
        front_cut_a = front_cut_b = end_cut_a = end_cut_b = 0

        while True:
            total_length = (
                len_a + len_b - front_cut_a - front_cut_b - end_cut_a - end_cut_b
            )
            if total_length <= max_num_tokens:
                break

            if len_a - front_cut_a - end_cut_a > len_b - front_cut_b - end_cut_b:
                if np.random.rand() < 0.5:
                    front_cut_a += 1
                else:
                    end_cut_a += 1
            else:
                if np.random.rand() < 0.5:
                    front_cut_b += 1
                else:
                    end_cut_b += 1

        # calculate ds indices as well as offsets and return
        truncated_sent_a = self._cut_sentence(sent_a, front_cut_a, end_cut_a)
        truncated_sent_b = self._cut_sentence(sent_b, front_cut_b, end_cut_b)
        return truncated_sent_a, truncated_sent_b

    def _cut_sentence(self, sent, front_cut, end_cut):
        """
        Cut a sentence based on the numbers of tokens to be cut from beginning and end
        Represent the sentence as dataset idx and return
        """
        start_ds_idx, end_ds_idx, offset = sent[0], sent[-1], 0
        target_len = sum(self.dataset.sizes[sent]) - front_cut - end_cut
        while front_cut > 0:
            if self.dataset.sizes[start_ds_idx] > front_cut:
                offset += front_cut
                break
            else:
                front_cut -= self.dataset.sizes[start_ds_idx]
                start_ds_idx += 1
        while end_cut > 0:
            if self.dataset.sizes[end_ds_idx] > end_cut:
                break
            else:
                end_cut -= self.dataset.sizes[end_ds_idx]
                end_ds_idx -= 1
        return start_ds_idx, offset, end_ds_idx, target_len

    def _fetch_block(self, start_ds_idx, offset, end_ds_idx, length):
        """
        Fetch a block of tokens based on its dataset idx
        """
        buffer = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )
        s, e = offset, offset + length
        return buffer[s:e]

    def __getitem__(self, index):
        block1, block2, next_sent_label = self.sent_pairs[index]
        block1 = self._fetch_block(*block1)
        block2 = self._fetch_block(*block2)
        return block1, block2, next_sent_label

    def __len__(self):
        return len(self.sizes)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        prefetch_idx = set()
        for index in indices:
            for block1, block2, _ in [self.sent_pairs[index]]:
                for ds_idx in range(block1[0], block1[2] + 1):
                    prefetch_idx.add(ds_idx)
                for ds_idx in range(block2[0], block2[2] + 1):
                    prefetch_idx.add(ds_idx)
        self.dataset.prefetch(prefetch_idx)
