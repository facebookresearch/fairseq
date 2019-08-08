# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch

from fairseq.data import FairseqDataset, plasma_utils


class TokenBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
    """

    def __init__(
        self, dataset, sizes, block_size, pad, eos, break_mode=None,
        include_targets=False, document_sep_len=1,
    ):
        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.include_targets = include_targets
        slice_indices = []

        assert len(dataset) == len(sizes)
        assert len(dataset) > 0
        sizes = np.array(sizes, dtype=int)
        if break_mode is None or break_mode == 'none':
            total_size = sum(sizes)
            length = math.ceil(total_size / block_size)

            def block_at(i):
                start = i * block_size
                end = min(start + block_size, total_size)
                return (start, end)

            slice_indices = [block_at(i) for i in range(length)]
        elif break_mode == 'complete':
            tok_idx = 0
            sz_idx = 0
            curr_size = 0
            while sz_idx < len(sizes):
                if curr_size + sizes[sz_idx] <= block_size or curr_size == 0:
                    curr_size += sizes[sz_idx]
                    sz_idx += 1
                else:
                    slice_indices.append((tok_idx, tok_idx + curr_size))
                    tok_idx += curr_size
                    curr_size = 0
            if curr_size > 0:
                slice_indices.append((tok_idx, tok_idx + curr_size))
        elif break_mode == 'complete_doc':
            tok_idx = 0
            sz_idx = 0
            curr_size = 0
            while sz_idx < len(sizes):
                if (
                    (curr_size + sizes[sz_idx] <= block_size or curr_size == 0)
                    # an empty sentence indicates end-of-document:
                    and sizes[sz_idx] != document_sep_len
                ):
                    curr_size += sizes[sz_idx]
                    sz_idx += 1
                else:
                    slice_indices.append((tok_idx, tok_idx + curr_size))
                    tok_idx += curr_size
                    curr_size = 0
                    if sizes[sz_idx] == document_sep_len:
                        tok_idx += sizes[sz_idx]
                        sz_idx += 1
            if curr_size > 0:
                slice_indices.append((tok_idx, tok_idx + curr_size))
        elif break_mode == 'eos':
            slice_indices = np.empty((len(sizes), 2), dtype=int)
            if not torch.is_tensor(sizes):
                sizes = torch.tensor(sizes)
            cumsum = torch.cumsum(sizes, dim=0)
            slice_indices[0] = [0, sizes[0]]
            if len(cumsum) > 1:
                slice_indices[1:] = cumsum.unfold(0, 2, 1)
        else:
            raise ValueError('Invalid break_mode: ' + break_mode)

        slice_indices = np.array(slice_indices, dtype=int)
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices
        if break_mode == 'eos':
            # much faster version for eos break mode
            block_to_dataset_index = np.stack(
                [
                    np.arange(len(sizes)),  # starting index in dataset
                    np.zeros(len(sizes), dtype=np.long),  # starting offset within starting index
                    np.arange(len(sizes))  # ending index in dataset
                ],
                1,
            )
        else:
            ds = DatasetSearcher(sizes)
            block_to_dataset_index = np.empty((len(slice_indices), 3), dtype=int)
            for i, (s, e) in enumerate(slice_indices):
                ds.seek(s)
                start_ds_idx = ds.current_index
                start_offset = ds.current_offset
                if e <= s:
                    continue
                ds.seek(e - 1)
                end_ds_idx = ds.current_index
                block_to_dataset_index[i] = (
                    start_ds_idx,  # starting index in dataset
                    start_offset,  # starting offset within starting index
                    end_ds_idx,  # ending index in dataset
                )

        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        buffer = torch.cat([
            self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)
        ])
        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length
        item = buffer[s:e]

        if self.include_targets:
            # *target* is the original sentence (=item)
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            if s == 0:
                source = torch.cat([item.new([self.eos]), buffer[0:e - 1]])
                past_target = torch.cat([item.new([self.pad, self.eos]), buffer[0:e - 2]])
            else:
                source = buffer[s - 1:e - 1]
                if s == 1:
                    past_target = torch.cat([item.new([self.eos]), buffer[0:e - 2]])
                else:
                    past_target = buffer[s - 2:e - 2]

            return source, item, past_target
        return item

    def __len__(self):
        return len(self.slice_indices)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.dataset.prefetch({
            ds_idx
            for index in indices
            for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
            for ds_idx in range(start_ds_idx, end_ds_idx + 1)
        })


class DatasetSearcher(object):
    """Helper for mapping "flat" indices to indices and offsets in an
    underlying dataset."""

    def __init__(self, sizes):
        self.sizes = sizes
        self.reset()

    def reset(self):
        self.current_index = 0  # index in underlying dataset
        self.current_offset = 0  # offset within current index in underlying dataset
        self.current_i = 0  # "flat" index

    def seek(self, i):
        assert i >= 0
        if i < self.current_i:
            self.reset()
        if i > self.current_i:
            to_consume = i - self.current_i
            remaining = self.sizes[self.current_index] - self.current_offset
            if remaining > to_consume:
                self.current_offset += to_consume
                self.current_i += to_consume
            else:
                self.current_i += remaining
                self.current_index += 1
                self.current_offset = 0
                self.seek(i)
        assert self.current_i == i
