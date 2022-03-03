# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset, data_utils
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


class AddTargetDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        pad,
        eos,
        batch_targets,
        process_label=None,
        label_len_fn=None,
        add_to_input=False,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.label_len_fn = label_len_fn
        self.add_to_input = add_to_input
        self.text_compressor = TextCompressor(level=text_compression_level)

    def get_label(self, index, process_fn=None):
        lbl = self.labels[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index, process_fn=self.process_label)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = self.label_len_fn(self.get_label(index))
        return sz, own_sz

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]

        if self.add_to_input:
            eos = torch.LongTensor([self.eos])
            prev_output_tokens = [torch.cat([eos, t], axis=-1) for t in target]
            target = [torch.cat([t, eos], axis=-1) for t in target]
            collated["net_input"]["prev_output_tokens"] = prev_output_tokens

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()
            collated["net_input"]["prev_output_tokens"] = data_utils.collate_tokens(
                collated["net_input"]["prev_output_tokens"],
                pad_idx=self.pad,
                left_pad=False,
            )
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["target"] = target
        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
