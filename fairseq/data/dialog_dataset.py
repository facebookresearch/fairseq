# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
# Vineet Kumar @ sioom: This file is a copy of language_pair_dataset.py,
# with changes made for the transaction bot implementation

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate(
    dlgs, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(dlgs) == 0:
        return {}

    def one_seq_per_dlg(key, seq_num_in_dlgs):
        one_seq_per_dlg = []
        for dlg in dlgs:
            try:
                one_seq_per_dlg.append(dlg[key][seq_num_in_dlgs])
            except IndexError:
                one_seq_per_dlg.append(dlg[key][0].new([eos_idx]))\
                      if dlg[key][0][-1] == eos_idx\
                      else one_seq_per_dlg.append(dlg[key][0].new([]))
        return one_seq_per_dlg

    def merge(key, seq_num_in_dlgs, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            one_seq_per_dlg(key, seq_num_in_dlgs), pad_idx, eos_idx,
            left_pad, move_eos_to_beginning,
        )

    batch = []
    dlg_ids = torch.LongTensor([dlg['dlg_id'] for dlg in dlgs])
    max_num_seqs_in_dlgs = max([len(dlg['source']) for dlg in dlgs])
    assert max_num_seqs_in_dlgs == max([len(dlg['target']) for dlg in dlgs])
    for seq_num_in_dlgs in range(max_num_seqs_in_dlgs):
        src_tokens = merge('source', seq_num_in_dlgs, left_pad=left_pad_source)
        # sort by descending source length
        src_lengths = torch.LongTensor([seq.numel() for seq in
                                       one_seq_per_dlg
                                       ('source', seq_num_in_dlgs)])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        dlg_ids_sorted = dlg_ids.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        prev_output_tokens = None
        target = None
        if dlgs[0].get('target', None) is not None:
            target = merge('target', seq_num_in_dlgs, left_pad=left_pad_target)
            target = target.index_select(0, sort_order)
            ntokens = sum(seq.numel() for seq in
                          one_seq_per_dlg('target', seq_num_in_dlgs))

            if input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                prev_output_tokens = merge(
                    'target',
                    seq_num_in_dlgs,
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens =\
                    prev_output_tokens.index_select(0, sort_order)
        else:
            ntokens = sum(seq.numel() for seq in
                          one_seq_per_dlg('source', seq_num_in_dlgs))

        batch.append({
            'id': dlg_ids_sorted,
            'nsentences': len(dlgs),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'sort_order': sort_order,
                'start_dlg': True if seq_num_in_dlgs == 0 else False
            },
            'target': target,
        })
        if prev_output_tokens is not None:
            batch[seq_num_in_dlgs]['net_input']['prev_output_tokens'] =\
                    prev_output_tokens
    return batch


class dialogDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        dialog_index (List[int]): maps dialog_id to the first sequence_id
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False,
        append_eos_to_target=False, dialog_index=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.dialog_index = dialog_index

    def _seq_ids(self, dlg_id):
        # Return sequences that belong to the dialog
        return [seq_id for seq_id in range(self.dialog_index[dlg_id],
                self.dialog_index[dlg_id+1]
                if dlg_id+1 != len(self.dialog_index)
                else len(self.src))]

    def __getitem__(self, dlg_id):
        src_items = []
        tgt_items = [] if self.tgt is not None else None
        for seq_id in self._seq_ids(dlg_id):
            src_items.append(self.src[seq_id])
            tgt_items.append(self.tgt[seq_id]) \
                if self.tgt is not None else None

        # Append EOS to end of tgt sentence if it does not have an EOS and
        # remove EOS from end of src sentence if it exists. This is useful when
        # we use existing datasets for opposite directions i.e., when we want
        # to use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target and self.tgt:
            tgt_items1 = tgt_items
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            tgt_items = [torch.cat([tgt_item, torch.LongTensor([eos])])
                         if tgt_item[-1] != eos else tgt_item
                         for tgt_item in tgt_items1]

        if self.remove_eos_from_source:
            src_items1 = src_items
            eos = self.src_dict.eos()
            src_items = [src_item[:-1] if src_item[-1] == eos else src_item
                         for src_item in src_items1]

        return {
            'dlg_id': dlg_id,
            'source': src_items,
            'target': tgt_items,
        }

    def __len__(self):
        return len(self.dialog_index)

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
                   appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will
                  appear on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def _num_tokens_src_tgt_dialogs(self, dlg_id):
        """Return the number of tokens in the dialog. This value is used to
        enforce ``--max-tokens`` during batching."""
        src_dialog_size = tgt_dialog_size = 0
        for seq_id in self._seq_ids(dlg_id):
            src_dialog_size += self.src_sizes[seq_id]
            tgt_dialog_size += self.tgt_sizes[seq_id] \
                if self.tgt_sizes is not None else 0
        return src_dialog_size, tgt_dialog_size

    def num_tokens(self, index):
        """Return the number of tokens in a dialog. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self._num_tokens_src_tgt_dialogs(index))

    def size(self, index):
        """Return an dialog's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self._num_tokens_src_tgt_dialogs(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))
        # In Translation application, sequences are ordered from those with
        # least # of tokens to those with max # of tokens. Doesn't make sense
        # to order dialogs based on the # of tokens they have. No other way of
        # ordering dialog indices makes sense, so ordering is not implemented

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False)
                 or self.tgt is None)
        )

    def prefetch(self, dlg_ids):
        seq_ids = [seq_id for dlg_id in dlg_ids
                   for seq_id in self._seq_ids(dlg_id)]
        self.src.prefetch(seq_ids)
        if self.tgt is not None:
            self.tgt.prefetch(seq_ids)
