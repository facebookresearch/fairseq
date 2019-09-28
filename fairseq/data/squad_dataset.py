# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset
from typing import List

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span[0] + doc_span[1] - 1
        if position < doc_span[0]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span[0]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span[1]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def collate(samples, pad_idx):
    if len(samples) == 0:
        return {}
    if not isinstance(samples[0], dict):
        samples = [s for sample in samples for s in sample]
    target_len = len(samples[0]['target'])
    target = [torch.stack([s['target'][i] for s in samples], dim=0) for i in range(target_len)]
    actual_txt = [s['actual_txt'] for s in samples]
    idx_map = [s['idx_map'] for s in samples]
    token_is_max_context = [s['token_is_max_context'] for s in samples]
    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(len(s['text']) for s in samples),
        'net_input': {
            'text': data_utils.collate_tokens(
                [s['text'] for s in samples], pad_idx, left_pad=False,
            ),
            'paragraph_mask': data_utils.collate_tokens([s['paragraph_mask'] for s in samples], pad_idx,  left_pad=False),
            'segment': data_utils.collate_tokens(
                [s['segment'] for s in samples], pad_idx, left_pad=False,
            ),
        },
        'target': target,
        'nsentences': len(samples),
        'actual_txt': actual_txt,
        'idx_map': idx_map,
        'possible_sentences': sum(int(s['target'][0] == 0) for s in samples),
        'squad_ids': [s['squad_ids'] for s in samples],
        'token_is_max_context':token_is_max_context
    }


class SquadDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset1, dataset2, 
                       labels, ids, actual_txt, 
                       idx_map,sizes1, sizes2, 
                       dictionary, stride,
                       max_length, max_query_length):
        self.dataset1, self.dataset2 = dataset1, dataset2
        self.sizes1, self.sizes2 = np.array(sizes1), np.array(sizes2)
        self.labels = np.array(labels)
        self.ids = ids
        self.actual_txt = actual_txt
        self.idx_map = idx_map
        self.vocab = dictionary
        self.shuffle = False
        self.stride = stride
        self.max_length = max_length
        self.max_query_length = max_query_length

    def __getitem__(self, index):
        paragraph = self.dataset1[index]
        question = self.dataset2[index]
        lbl = self.labels[index]
        actual_txt = self.actual_txt[index]
        idx_map = [int(ii) for ii in self.idx_map[index]]
        if question.size(0) > self.max_query_length:
            question = question[:self.max_query_length]
        question_len = question.size(0) + 2  # account for cls + sep
        start_offset = 0
        doc_spans_text = []
        doc_spans = []
        max_tokens_for_doc = self.max_length - len(question) - 3
        while start_offset < len(paragraph):
            length = len(paragraph) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans_text.append(paragraph[start_offset: start_offset+length])
            doc_spans.append((start_offset, length))
            if start_offset + length == len(paragraph):
                break
            start_offset = start_offset +  min(length, self.stride)

        if len(lbl) == 0:
            s, e = -1, -1
        else:
            s, e = lbl
            assert e >= s
        res  = []
        for span_idx, span in enumerate(doc_spans_text):
            span_idx_map = []
            doc_start = doc_spans[span_idx][0]
            doc_end = doc_spans[span_idx][0] + doc_spans[span_idx][1] - 1
            span_idx_map = idx_map[doc_start : doc_end+1]
            out_of_span = False
            if not (s >= doc_start and e <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                start_position = s - doc_start + question_len
                end_position = e - doc_start + question_len
            start_position = torch.LongTensor([start_position])
            end_position = torch.LongTensor([end_position])
            text, seg = self._join_sents(question, span)
            paragraph_mask = torch.zeros(text.shape).byte()
            paragraph_mask[question_len : -1] = 1
            target = (start_position, end_position)
            token_is_max_context = [0] * question_len
            for j in range(doc_spans[span_idx][1]):
                split_token_index = doc_spans[span_idx][0] + j
                is_max_context = _check_is_max_context(doc_spans, span_idx, split_token_index)
                token_is_max_context.append(int(is_max_context))
            res.append({'id': index, 'text': text, 'segment': seg, 'target': target, 'paragraph_mask': paragraph_mask,
                'squad_ids': self.ids[index], 'actual_txt':actual_txt, 'idx_map':torch.LongTensor(span_idx_map), 'token_is_max_context':torch.LongTensor(token_is_max_context)})
        return res

    def _join_sents(self, sent1, sent2):
        cls = sent1.new_full((1,), self.vocab.cls())
        sep = sent1.new_full((1,), self.vocab.sep())
        sent1 = torch.cat([cls, sent1, sep])
        sent2 = torch.cat([sent2, sep])
        text = torch.cat([sent1, sent2])
        segment1 = torch.zeros(sent1.size(0))
        segment2 = torch.ones(sent2.size(0))
        segment = torch.cat([segment1, segment2])

        return text, segment

    def __len__(self):
        return len(self.dataset1)

    def collater(self, samples):
        return collate(samples, self.vocab.pad())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        sent1 = self.vocab.dummy_sentence(tgt_len + 2)
        sent2 = self.vocab.dummy_sentence(tgt_len + 2)

        sent1[sent1.eq(self.vocab.unk())] = 66
        sent2[sent2.eq(self.vocab.unk())] = 66
        text, segment = self._join_sents(sent1, sent2)

        paragraph_mask = torch.zeros(text.shape).byte()
        paragraph_mask[sent2.numel():] = 1
        
        target = (torch.tensor([self.vocab.pad()]), torch.tensor([self.vocab.pad()]))
        idx_map = [self.vocab.pad()]
        token_is_max_context = [0]
        return self.collater([
            {'id': i, 'text': text, 'target': target, 'segment': segment, 'paragraph_mask': paragraph_mask, 'squad_ids': 0, 'actual_txt':'dummy', 'idx_map':idx_map,'token_is_max_context':token_is_max_context}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes1[index] + self.sizes2[index] + 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes1[index] + self.sizes2[index] + 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            return  np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        indices = indices[np.argsort(self.sizes1[indices], kind='mergesort')]
        return indices[np.argsort(self.sizes2[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.dataset1.prefetch(indices)
        self.dataset2.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
                hasattr(self.dataset1, 'supports_prefetch')
                and self.dataset1.supports_prefetch
                and hasattr(self.dataset2, 'supports_prefetch')
                and self.dataset2.supports_prefetch
        )
