# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from torch.utils.data.dataloader import default_collate

from . import FairseqDataset, MonolingualDataset


class MonolingualDatasetsWithLabels(FairseqDataset):
    """
    A wrapper around several :class:`~fairseq.data.MonolingualDataset` objects with
    optional labels.

    Inputs are concatenated and collated batches will include *segment_labels*
    indicating the source of each token.

    Args:
        input_datasets (List[~fairseq.data.MonolingualDataset]): datasets to wrap
        labelss (List[int]): labels
        shuffle (bool, optional): shuffle data
        add_bos_token (bool, optional): add beginning of sentence
        concat_inputs (bool, optional): concatenate inputs (default: True)
        separator_token (int, optional): token idx to be used as separator token.
        cls_token (int, optional): token idx to be used as cls token used for masked_lm model.
    """

    def __init__(
        self,
        input_datasets,
        labels,
        shuffle=True,
        add_bos_token=False,
        concat_inputs=True,
        separator_token=None,
        cls_token=None,
    ):
        assert len(input_datasets) > 0
        assert all(isinstance(ds, MonolingualDataset) for ds in input_datasets)

        self.input_datasets = input_datasets
        self.labels = labels
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token
        self.concat_inputs = concat_inputs
        self.separator_token = separator_token
        self.cls_token = cls_token

        if separator_token is not None or self.cls_token is not None:
            assert concat_inputs

        if add_bos_token:
            raise NotImplementedError

        self.sizes = sum(ds.sizes for ds in input_datasets)

    def __getitem__(self, index):
        return {
            'id': index,
            'input': [ds[index]['source'] for ds in self.input_datasets],
            'segments': [
                torch.full_like(ds[index]['source'], seg_idx)
                for seg_idx, ds in enumerate(self.input_datasets)
            ],
            'labels': torch.tensor(self.labels[index])
        }

    def __len__(self):
        return len(self.input_datasets[0])

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        if len(samples) == 0:
            return {}

        if self.concat_inputs:
            # reuse collation from MonolingualDataset
            def concat_with_seperator(s, is_segment=False):
                if self.separator_token is None:
                    source = torch.cat(s)
                else:
                    ss = []
                    for i in range(0, len(s)):
                        separator = self.separator_token if not is_segment else s[i][0].item()
                        ss.append(s[i])
                        ss.append(s[i].new([separator]))
                    source = torch.cat(ss)
                return source

            def prepend_cls(s, is_segment=False):
                if self.cls_token is None:
                    return s
                cls_token = self.cls_token if not is_segment else s[0].item()
                return torch.cat([s.new([cls_token]), s])

            sample = self.input_datasets[0].collater([
                {
                    'id': s['id'],
                    'source': prepend_cls(concat_with_seperator(s['input'])),
                    'target': None,
                }
                for s in samples
            ])

            # collate segment tokens
            sample['net_input']['segment_labels'] = self.input_datasets[0].collater([
                {
                    'id': s['id'],
                    'source': prepend_cls(
                        concat_with_seperator(s['segments'], is_segment=True),
                        is_segment=True,
                    ),
                    'target': None
                }
                for s in samples
            ])['net_input']['src_tokens']
        else:
            # reuse collation from MonolingualDataset
            num_inputs = len(samples[0]['input'])
            sample = self.input_datasets[0].collater([
                {'id': s['id'], 'source': s['input'][0], 'target': None}
                for s in samples
            ])
            for k in sample['net_input'].keys():
                sample['net_input'][k] = [sample['net_input'][k]]
            for i in range(1, num_inputs):
                sample_i = self.input_datasets[i].collater([
                    {'id': s['id'], 'source': s['input'][i], 'target': None}
                    for s in samples
                ])
                for k in sample_i['net_input'].keys():
                    sample['net_input'][k].append(sample_i['net_input'][k])

        # reuse each label datasets collater
        sample['target'] = default_collate([s['labels'] for s in samples])
        return sample

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
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return all(
            getattr(ds, 'supports_prefetch', False)
            for ds in self.input_datasets
        )

    def prefetch(self, indices):
        for ds in self.input_datasets:
            ds.prefetch(indices)
