# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset


class OddOneOutDataset(FairseqDataset):

    def __init__(
        self,
        dataset,
        sizes,
        vocab,
        max_tokens,
        short_item_prob=0.1,
        document_sep_len=1,
    ):
        self.dataset = dataset
        self.sizes = sizes
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.short_item_prob = short_item_prob
        self.document_sep_len = document_sep_len

        assert len(dataset) == len(sizes)
        assert sizes.min() == document_sep_len, \
            'Documents are expected to be separated by a single blank line (</s>).'

        # group sentences into docs
        document_seps = torch.from_numpy(sizes).eq(document_sep_len)

        # exclude empty documents (i.e., two or more adjacent document seps)
        # 0 0 0 1 0 0 1 1 0
        # ->
        # 0 0 0 0 0 0 1 0
        empty_docs = document_seps[:-1] * document_seps[1:]
        document_seps[1:] -= empty_docs

        endpoints = document_seps.nonzero().view(-1)
        self.doc_sent_index = np.split(np.arange(len(sizes)), endpoints)

        # dynamically maintain a 50/50 class balance for Odd-One-Out task
        self._ooo_class_balance = [0, 0]

        self._prefetched_ids = None

    def sent_ids(self, doc_id):
        sent_ids = self.doc_sent_index[doc_id]
        # exclude any eos at the beginning of each doc
        while self.sizes[sent_ids[0]] == self.document_sep_len:
            sent_ids = sent_ids[1:]
        return sent_ids

    def __getitem__(self, index):
        item = []  # collection of sentences, maybe longer than max_tokens
        item_len = [0]
        doc_start = [0]
        ooo_endpoints = []
        ooo_endpoint_labels = []

        def get_rand_doc_id(doc_to_exclude):
            if self._prefetched_ids is None:
                # generate in range [0, num_docs - 2] so that we can exclude doc_to_exclude
                doc_id = torch.randint(0, len(self.doc_sent_index) - 1, (1, ))
                if doc_id >= doc_to_exclude:
                    doc_id += 1  # exclude doc_to_exclude
            else:
                # only select from prefetched docs
                pf_id = torch.randint(0, len(self._prefetched_ids) - 1, (1, ))
                doc_id = self._prefetched_ids[pf_id]
                if doc_id == doc_to_exclude:
                    pf_id = len(self._prefetched_ids) - 1
                    doc_id = self._prefetched_ids[pf_id]
            return doc_id

        def get_rand_sent_id(doc_to_exclude):
            doc_id = get_rand_doc_id(doc_to_exclude)
            sent_ids = self.sent_ids(doc_id)
            if len(sent_ids) == 1:
                return sent_ids[0]
            else:
                # we have multiple sentence, pick a random one, excluding the first
                i = torch.randint(1, len(sent_ids), (1, )).item()
                return sent_ids[i]

        def add_sentence(sent_id, ooo_label, prepend_eos=False):
            toks = self.dataset[sent_id][1]
            assert toks[-1].item() == self.vocab.eos()
            if prepend_eos:
                item.append(toks.new([self.vocab.eos()]))
                item_len[0] += 1
            item.append(toks)
            item_len[0] += len(toks)
            ooo_endpoints.append((doc_start[0], item_len[0] - 1))
            ooo_endpoint_labels.append(ooo_label)

        def add_doc(doc_id, sent_ids=None):
            if sent_ids is None:
                sent_ids = self.sent_ids(doc_id)

            # always add first sentence, starting with eos to indicate doc boundary
            add_sentence(sent_ids[0], ooo_label=0, prepend_eos=True)

            if len(sent_ids) > 1:
                # If there are multiple sentences in the doc, maybe replace one.
                # Also try to maintain an even balance between OOO classes.
                if self._ooo_class_balance[0] >= self._ooo_class_balance[1]:
                    ooo_label = 1
                    id_to_replace = torch.randint(1, len(sent_ids), (1, )).item()
                else:
                    ooo_label = 0
                    id_to_replace = -1
                self._ooo_class_balance[ooo_label] += 1

                for j, sent_id in enumerate(sent_ids[1:], start=1):
                    if j == id_to_replace:
                        add_sentence(get_rand_sent_id(doc_to_exclude=doc_id), ooo_label)
                    else:
                        add_sentence(sent_id, ooo_label=0)

            # next doc (if there is one) starts here
            doc_start[0] = item_len[0]

        def get_rand_doc_subset(sent_ids):
            # return a random subset of the document with at least max_tokens
            rev_lengths = torch.tensor(
                [self.sizes[sent_id] for sent_id in reversed(sent_ids)],
                dtype=torch.long
            )
            rev_cumsum = torch.cumsum(rev_lengths, 0)
            if rev_cumsum[-1] >= self.max_tokens:
                # the document is very long, pick a random start point
                rev_max_start_sent_idx = (rev_cumsum >= self.max_tokens).nonzero()[0].item()
                max_start_sent_idx = len(sent_ids) - rev_max_start_sent_idx - 1
                start_sent_idx = torch.randint(0, max_start_sent_idx + 1, (1, )).item()

                # pick the min number of sentences to reach max_tokens
                subset_size = 0
                end_sent_idx = start_sent_idx
                while subset_size < self.max_tokens:
                    subset_size += self.sizes[sent_ids[end_sent_idx]]
                    end_sent_idx += 1
                sent_ids = sent_ids[start_sent_idx:end_sent_idx]

                assert sum(self.sizes[sent_id] for sent_id in sent_ids) >= self.max_tokens
            else:
                # the document is smaller than max_tokens, return the whole thing
                pass
            return sent_ids

        sent_ids = self.sent_ids(index)
        if torch.rand(1).item() < self.short_item_prob:
            # make a short item with 1 or 2 sentences only
            if len(sent_ids) == 1:
                add_doc(index, sent_ids)
            else:
                # we only need two sentences, so pick a random start position
                # that supports this
                start_pos = torch.randint(0, len(sent_ids) - 1, (1, )).item()
                add_doc(index, sent_ids[start_pos:start_pos + 2])
        else:
            # if the document is very long, get a random subset that provides
            # at least max_tokens
            sent_ids = get_rand_doc_subset(sent_ids)

            add_doc(index, sent_ids)

            # if the document wasn't long enough, pad with random docs
            while item_len[0] < self.max_tokens:
                doc_id = get_rand_doc_id(doc_to_exclude=index)
                add_doc(doc_id)

        # concatenate document and truncate if it's too long
        item = torch.cat(item)[:self.max_tokens]
        ooo_endpoints = torch.tensor(ooo_endpoints)
        ooo_endpoint_labels = torch.tensor(ooo_endpoint_labels)
        if ooo_endpoints.max().item() >= self.max_tokens:
            valid_endpoints = (ooo_endpoints.max(dim=1)[0] < self.max_tokens)
            ooo_endpoints = ooo_endpoints[valid_endpoints]
            ooo_endpoint_labels = ooo_endpoint_labels[valid_endpoints]

        return {
            'id': index,
            'source': item,
            'target': item,
            'ooo_endpoints': ooo_endpoints,
            'ooo_endpoint_labels': ooo_endpoint_labels,
        }

    def __len__(self):
        return len(self.doc_sent_index)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        pad_idx = self.vocab.pad()
        eos_idx = self.vocab.eos()

        src_tokens = data_utils.collate_tokens(
            [s['source'] for s in samples], pad_idx, eos_idx, left_pad=False,
        )

        seq_len = src_tokens.size(1)
        ooo_endpoints = torch.cat([
            s['ooo_endpoints'].view(-1) + (seq_len * i)
            for i, s in enumerate(samples)
        ])
        ooo_endpoint_labels = torch.cat([
            s['ooo_endpoint_labels'].view(-1) for i, s in enumerate(samples)
        ])

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'nsentences': len(samples),
            'ntokens': sum(len(s['source']) for s in samples),
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': torch.LongTensor([
                    s['source'].numel() for s in samples
                ]),
            },
            'target': src_tokens,
            'ooo_endpoints': ooo_endpoints,
            'ooo_endpoint_labels': ooo_endpoint_labels,
        }

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.max_tokens

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.max_tokens

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        self._prefetched_ids = indices
        self.dataset.prefetch({
            ds_idx
            for index in indices
            for ds_idx in self.doc_sent_index[index]
        })
