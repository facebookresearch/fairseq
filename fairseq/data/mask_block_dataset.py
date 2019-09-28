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


class ModifiedBlockPairDataset(torch.utils.data.Dataset):
    """Break a 1d tensor of tokens into sentence pair blocks for next sentence
       prediction as well as masked language model.
       High-level logics are:
       1. break input tensor to tensor blocks
       2. pair the blocks with 50% next sentence and 50% random sentence
       3. return paired blocks as well as related segment labels
    Args:
        tokens: 1d tensor of tokens to break into blocks
        block_size: maximum block size
        pad: pad index
        eos: eos index
        cls: cls index
        mask: mask index
        sep: sep index to separate blocks
    """

    def __init__(
        self,
        tokens,
        sizes,
        block_size,
        pad,
        class_positive,
        class_negative,
        sep,
        vocab,
        break_mode="doc",
        short_seq_prob=0,
    ):
        super().__init__()

        self.tokens = tokens
        self.total_size = len(tokens)
        self.pad = pad
        self.class_positive = class_positive
        self.class_negative = class_negative
        self.sep = sep
        self.vocab = vocab
        self.block_indices = []
        self.break_mode = break_mode

        if break_mode == "sentence":
            assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
            curr = 0
            for sz in sizes:
                if sz == 0:
                    continue
                self.block_indices.append((curr, curr + sz))
                curr += sz
            max_num_tokens = block_size - 3 # Account for [CLS], [SEP], [SEP]
            self.sent_pairs = []
            self.sizes = []
            target_seq_length = max_num_tokens
            if np.random.random() < short_seq_prob:
                target_seq_length = np.random.randint(2, max_num_tokens)
            current_chunk = []
            current_length = 0
            curr = 0
            while curr < len(self.block_indices):
                sent = self.block_indices[curr]
                current_chunk.append(sent)
                current_length = current_chunk[-1][1] - current_chunk[0][0]
                if curr == len(self.block_indices) - 1 or current_length >= target_seq_length:
                    if current_chunk:
                        a_end = 1
                        if len(current_chunk) > 2:
                            a_end = np.random.randint(1, len(current_chunk) - 1)
                        sent_a = current_chunk[:a_end]
                        sent_a = (sent_a[0][0], sent_a[-1][1])
                        next_sent_label = (
                            1 if np.random.rand() > 0.5 else 0
                        )
                        if len(current_chunk) == 1 or next_sent_label:
                            target_b_length = target_seq_length - (sent_a[1] - sent_a[0])
                            random_start = np.random.randint(0, len(self.block_indices) - len(current_chunk))
                            # avoid current chunks
                            # we do this just because we don't have document level segumentation
                            random_start = (
                                random_start + len(current_chunk)
                                if self.block_indices[random_start][1] > current_chunk[0][0]
                                else random_start
                            )
                            sent_b = []
                            for j in range(random_start, len(self.block_indices)):
                                sent_b = (
                                    (sent_b[0], self.block_indices[j][1])
                                    if sent_b else self.block_indices[j]
                                )
                                if self.block_indices[j][0] == current_chunk[0][0]:
                                    break
                                # length constraint
                                if sent_b[1] - sent_b[0] >= target_b_length:
                                    break
                            num_unused_segments = len(current_chunk) - a_end
                            curr -= num_unused_segments
                            next_sent_label = 1
                        else:
                            sent_b = current_chunk[a_end:]
                            sent_b = (sent_b[0][0], sent_b[-1][1])
                        sent_a, sent_b = self._truncate_sentences(sent_a, sent_b, max_num_tokens)
                        self.sent_pairs.append((sent_a, sent_b, next_sent_label))
                        if sent_a[0] >= sent_a[1] or sent_b[0] >= sent_b[1]:
                            print(sent_a, sent_b)
                        self.sizes.append(3 + sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0])
                    current_chunk = []
                curr += 1
        elif break_mode == "doc":
            #self.sent_pairs = []
            self.sizes = []
            self.sents = []
            assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
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
            max_num_tokens = block_size - 2 # Account for [CLS], [SEP], [SEP]
            for doc_id, doc in enumerate(self.block_indices):
                current_chunk = []
                current_length = 0
                curr = 0
                target_seq_length = max_num_tokens
                if np.random.random() < short_seq_prob:
                    target_seq_length = np.random.randint(2, max_num_tokens)
                while curr < len(doc):
                    sent = doc[curr]
                    if sent[1] - sent[0] <= max_num_tokens:
                        current_chunk.append(sent)
                        current_length = current_chunk[-1][1] - current_chunk[0][0]
                        if curr == len(doc) - 1 or current_length >= target_seq_length:
                            if current_length > max_num_tokens:
                                current_chunk = current_chunk[:-1]
                                curr -= 1
                            if len(current_chunk) > 0:
                                sent = (current_chunk[0][0], current_chunk[-1][1])
                                self.sents.append(sent)
                                self.sizes.append(sent[1] - sent[0] + 2)
                            current_chunk = []
                    curr += 1
        else:
            block_size -= 3
            block_size //= 2  # each block should have half of the block size since we are constructing block pair
            length = math.ceil(len(tokens) / block_size)

            def block_at(i):
                start = i * block_size
                end = min(start + block_size, len(tokens))
                return (start, end)

            self.block_indices = [block_at(i) for i in range(length)]

            self.sizes = np.array(
                # 2 block lengths + 1 cls token + 2 sep tokens
                # note: this is not accurate and larger than pairs including last block
                [block_size * 2 + 3] * len(self.block_indices)
            )

    def _truncate_sentences(self, sent_a, sent_b, max_num_tokens):
        while True:
            total_length = sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0]
            if total_length <= max_num_tokens:
                return sent_a, sent_b

            if sent_a[1] - sent_a[0] > sent_b[1] - sent_b[0]:
                sent_a = (
                    (sent_a[0]+1, sent_a[1])
                    if np.random.rand() < 0.5
                    else (sent_a[0], sent_a[1] - 1)
                )
            else:
                sent_b = (
                    (sent_b[0]+1, sent_b[1])
                    if np.random.rand() < 0.5
                    else (sent_b[0], sent_b[1] - 1)
                )

    def _rand_block_index(self, i):
        """select a random block index which is not given block or next
           block
        """
        idx = np.random.randint(len(self.block_indices) - 3)
        return idx if idx < i else idx + 2

    def _mask_block(self, sentence, mask_ratio=1):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        sentence = np.copy(sentence)
        target = np.copy(sentence)
        return sentence, target

    def __getitem__(self, index):
        if self.break_mode == "sentence":
            block1, block2, next_sent_label = self.sent_pairs[index]
        elif self.break_mode == "doc":
            block1 =  self.sents[index]
            next_sent_label=1
        else:
            next_sent_label = (
                1 if np.random.rand() > 0.5 else 0
            )
            block1 = self.block_indices[index]
            if next_sent_label:
                block2 = self.block_indices[self._rand_block_index(index)]
            elif index == len(self.block_indices) - 1:
                next_sent_label = 1
                block2 = self.block_indices[self._rand_block_index(index)]
            else:
                block2 = self.block_indices[index+1]

        masked_blk1, masked_tgt1 = self._mask_block(self.tokens[block1[0]:block1[1]])
        #masked_blk2, masked_tgt2 = self._mask_block(self.tokens[block2[0]:block2[1]])

        #if next_sent_label:
        #    cls = self.class_positive
        #else:
        cls = self.class_negative

        item1 = np.concatenate(
            [
                [cls],
                masked_blk1,
                [self.sep],
            ]

        )
        """item2 = np.concatenate(
            [
                masked_blk2,
                [self.sep],
            ]

        )"""
        target1 = np.concatenate([[self.pad], masked_tgt1, [self.pad]])
        #target2 = np.concatenate([masked_tgt2, [self.pad]])
        
        seg0 = np.full((1), 2)
        seg1 = np.zeros((block1[1] - block1[0]) + 1)  # block + 1 sep + 1 cls
        #seg2 = np.ones((block2[1] - block2[0]) + 1)  # block + 1 sep

        #item = np.concatenate([item1, item2])
        seg = np.concatenate([seg0, seg1])#, seg2])
        #target = np.concatenate([target1, target2])

        return torch.LongTensor(item1), torch.LongTensor(seg), torch.LongTensor(target1), next_sent_label

    def __len__(self):
        return len(self.sizes)


class ModifiedBertDataset(FairseqDataset):
    """
    A wrapper around BlockPairDataset for BERT data.
    Args:
        dataset (BlockPairDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, sizes, 
            vocab, shuffle, seed, 
            mask_ratio, lower=None, 
            upper=None, geometric_p=-1):

        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.mask_ratio = mask_ratio
        self.lower = lower
        self.upper = upper
        self.lens = list(range(self.lower, self.upper + 1)) if self.lower is not None and self.upper is not None else None
        self.p = geometric_p
        self.len_distrib = [self.p * (1-self.p)**(i - self.lower) for i in range(self.lower, self.upper + 1)] if self.p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib] if self.len_distrib is not None else None

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            (
                source, segment_labels, lm_target, sentence_target
            ) = self.dataset[index]
            enc, dec = None, None
            if self.p > 0:
                mask_spans, not_mask = self.span_based_mask(source, lm_target)
                enc, dec = self.make_mask(mask_spans, not_mask)

            
        return {
            'id': index,
            'source': source,
            'segment_labels': segment_labels,
            'lm_target': lm_target,
            'sentence_target': sentence_target,
            'enc': enc,
            'dec': dec
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
        lm_target = merge('lm_target')
        enc = samples[0]['enc']
        
        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'ntokens': sum(len(s['source']) for s in samples),
            'net_input': {
                'src_tokens': merge('source'),
                'segment_labels': merge('segment_labels'),
                'target': lm_target if not self.p > 0 else None,
                'enc_mask': torch.stack([s['enc'] for s in samples]) if enc is not None else None,
                'dec_mask':torch.stack([s['dec'] for s in samples]) if enc is not None else None,
            },
            'target': lm_target if self.p>0 else None,
            'sentence_target': torch.LongTensor([s['sentence_target'] for s in samples]),
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
        sentence_target = 0
        bsz = num_tokens // tgt_len

        return self.collater([
            {
                'id': i,
                'source': source,
                'segment_labels': segment_labels,
                'lm_target': lm_target,
                'sentence_target': sentence_target,
                'enc': torch.empty(513, 513).fill_(0).byte() if self.p >0 else None,
                'dec': torch.empty(513, 513).fill_(1).byte() if self.p>0 else None
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
        order = [np.arange(len(self)), self.sizes]
        return np.lexsort(order)

    def get_word_piece_map(self, word):
        if self.vocab[word].startswith('##'):
            return True
        return False

    def get_word_start(self, sentence, anchor):
        word_piece_map = [self.get_word_piece_map(w) for w in sentence]
        left  = anchor
        while left > 0 and word_piece_map[left] == True:
            left -= 1
        return left

    def get_word_end(self, sentence, anchor):
        word_piece_map = [self.get_word_piece_map(w) for w in sentence]
        right = anchor + 1
        while right < len(sentence) and word_piece_map[right] == True:
            right += 1
        return right

    def span_based_mask(self, sentence, target):
        sent_length = len(target)
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = set()
        masks_span = []
        while len(mask) < mask_num:
            current_span = []
            anchor  = np.random.choice(sent_length)
            if anchor in mask:
                continue
            left_word_idx, right_word_idx = self.get_word_start(sentence, anchor), self.get_word_end(sentence, anchor)
            for i in range(left_word_idx, right_word_idx):
                if len(mask) >= mask_num:
                    break
                if i not in mask:
                    mask.add(i)
                    current_span.append(i)
            num_words = 1
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            next_start_idx = right_word_idx
            while num_words < span_len and next_start_idx < len(sentence) and len(mask) < mask_num:
                next_end_idx= self.get_word_end(sentence, next_start_idx)
                num_words += 1
                for i in range(next_start_idx, next_end_idx):
                    if len(mask) >= mask_num:
                        break
                    if i not in mask:
                        mask.add(i)
                        current_span.append(i)
            masks_span.append(current_span)
                
        not_mask = [i for i in range(sent_length) if i not in mask]
        target[not_mask] = self.vocab.pad()
        assert sum(len(j) for j in masks_span) == len(mask)
        return masks_span, not_mask
   
    def make_mask(self, masks_span, not_mask):
        total_span = len(masks_span)
        blockA = masks_span[:len(masks_span) // 2]
        blockB = masks_span[len(masks_span) // 2:]
        # switch all the block by one and add dummy position 0 to blockC
        blockA = sorted([i+1 for j in blockA for i in j])
        blockB = sorted([i+1 for j in blockB for i in j])
        blockC = [i+1 for i in not_mask]
        blockC.append(0)
        blockA_size, blockB_size, blockC_size = len(blockA), len(blockB), len(blockC)
        mask = torch.empty(513, 513).fill_(1).byte()

        # construct encoder mask
        enc_mask = mask.clone()
        for idx in blockC:
            enc_mask[idx, blockC] = 0
        block_B = blockC + blockB
        for idx_idx, idx in enumerate(blockB):
            enc_mask[idx, block_B[:blockC_size+idx_idx+1]] = 0
        block_A = blockC + blockA
        for idx_idx, idx in enumerate(blockA):
            enc_mask[idx, block_A[:blockC_size+idx_idx+1]] = 0

        # construct decoder mask
        dec_mask = mask.clone()
        for idx in blockC:
            dec_mask[idx, blockC] = 0
        block_B = blockC + blockA + blockB
        for idx_idx, idx in enumerate(blockB):
            dec_mask[idx, block_B[:blockC_size+blockA_size+idx_idx]] = 0
        block_A = blockC + blockB + blockA
        for idx_idx, idx in enumerate(blockA):
            dec_mask[idx, block_A[:blockC_size+blockB_size+idx_idx]] = 0
        
        return enc_mask, dec_mask

        

