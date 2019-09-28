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
    """Break a 1d tensor of tokens into sentence pair blocks for next sentence
       prediction as well as masked language model.
       High-level logics are:
       1. break input tensor to tensor blocks
       2. pair the blocks with 50% next sentence and 50% random sentence
       3. return paired blocks as well as related segment labels
    Args:
        tokens: 1d tensor of tokens to break into blocks
        max_block_size: maximum block size
    """

    def __init__(
        self,
        tokens,
        sentence_sizes,
        max_block_size,   #TODO currently ignoring max_block_size because the data is pre-segmented
    ):
        super().__init__()
        self.tokens = tokens
        self.block_indices = []

        assert sentence_sizes is not None and sum(sentence_sizes) == len(tokens), '{} != {}'.format(sum(sentence_sizes), len(tokens))

        self.docs = []
        doc = []
        curr = 0
        for sent in sentence_sizes:
            if (sent == 0) and (len(doc) > 0):
                self.docs.append(doc)
                doc = []
            else:
                doc.append((curr, curr + sent))
            curr += sent
        if len(doc) > 0:
            self.docs.append(doc)

        self.sizes = [doc[-1][1] - doc[0][0] for doc in self.docs]

    def __getitem__(self, index):
        return self.docs[index]

    def __len__(self):
        return len(self.docs)


class BertDataset(FairseqDataset):
    """
    A wrapper around BlockPairDataset for BERT data.
    Args:
        dataset (BlockPairDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, sizes, vocab, shuffle, seed, fix_seed, token_mask_ratio, token_noise_prob, token_clean_prob, sent_pos_mask_ratio, sent_pos_noise_prob, sent_pos_clean_prob):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.token_mask_ratio = token_mask_ratio
        self.token_noise_prob = token_noise_prob
        self.token_clean_prob = token_clean_prob
        self.sent_pos_mask_ratio = sent_pos_mask_ratio
        self.sent_pos_noise_prob = sent_pos_noise_prob
        self.sent_pos_clean_prob = sent_pos_clean_prob
        self.fix_seed = fix_seed

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            doc = self.dataset[index]
        
        sents = [self.dataset.tokens[sent_start:sent_end] for sent_start, sent_end in doc]
        sent_pos = list(range(2, len(sents)+2)) # 0 = pad, 1 = sent_pos mask
        source_sents, target_sents = zip(*[self._mask_block(sent, self.dataset.tokens, self.vocab.mask(), index, self.token_mask_ratio, self.token_noise_prob, self.token_clean_prob) for sent in sents])
        source_sent_pos, target_sent_pos = self._mask_block(sent_pos, sent_pos, 1, index, self.sent_pos_mask_ratio, self.sent_pos_noise_prob, self.sent_pos_clean_prob)
        
        max_sent_len = max(map(len, sents))
        source_sents = [np.concatenate([[source_sent_pos[i]], sent, [self.vocab.pad()] * (max_sent_len - len(sent))]) for i, sent in enumerate(source_sents)]
        target_sents = [np.concatenate([[target_sent_pos[i]], sent, [self.vocab.pad()] * (max_sent_len - len(sent))]) for i, sent in enumerate(target_sents)]
        
        source = np.stack(source_sents, axis=0)
        target = np.stack(target_sents, axis=0)
        
        return {
            'id': index,
            'source': torch.from_numpy(source).long(),
            'target': torch.from_numpy(target).long(),
        }

    def __len__(self):
        return len(self.dataset)

    def _collate(self, samples, pad_idx):

        if len(samples) == 0:
            return {}
        
        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)
        
        def merge(key):
            """Convert a list of n-order tensors into a padded n+1-order tensor."""
            values = [s[key] for s in samples]

            order = len(values[0].size())
            sizes = [max(v.size(d) for v in values) for d in range(order)]
            sizes = [len(values)] + sizes
            
            res = values[0].new(*sizes).fill_(pad_idx)
            for i, v in enumerate(values):
                copy_tensor(v, res[i][[slice(s) for s in v.size()]])
            return res
        
        source = merge('source')
        target = merge('target')

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'ntokens': source.ne(pad_idx).sum().item(),
            'net_input': {
                'src_tokens': source,
            },
            'target': target,
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
        sents = [self.vocab.dummy_sentence(tgt_len), self.vocab.dummy_sentence(tgt_len), self.vocab.dummy_sentence(tgt_len)]
        source = torch.stack(sents, dim=0)
        sent_pos = source.new(3, 1)
        sent_pos[0, 0] = 1
        sent_pos[1, 0] = 2
        sent_pos[2, 0] = 3
        source = torch.cat([sent_pos, source], dim=1)
        target = source
        bsz = 1
        
        return self.collater([
            {
                'id': i,
                'source': source,
                'target': target,
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
        _, _, order = zip(*sorted([(max(sent[1] - sent[0] for sent in doc), len(doc), i) for i, doc in enumerate(self.dataset.docs)]))
        return list(order)

    def _mask_block(self, sentence, tokens, mask_token, idx, mask_ratio, noise_prob, clean_prob):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        mask_thr = 1 - noise_prob - clean_prob
        noise_thr = mask_thr + noise_prob

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
                if rand < mask_thr:
                    sentence[i] = mask_token
                elif rand < noise_thr:
                    # sample random token according to input distribution
                    sentence[i] = np.random.choice(tokens)
            else:
                target[i] = self.vocab.pad()
        return sentence, target

