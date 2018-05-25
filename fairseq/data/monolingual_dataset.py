# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
import torch

from torch.utils.data import Dataset

from fairseq.data import TokenBlockDataset, Dictionary, LanguageDatasets
from fairseq.data.indexed_dataset import IndexedInMemoryDataset
from fairseq.data.data_utils import fmt_path, collate_tokens


def collate(samples, pad_idx, eos_idx, has_target):
    if len(samples) == 0:
        return {}

    def merge(key):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad=False, move_eos_to_beginning=False,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    # language models only have a decoder which is not padding-aware, so don't left pad for them
    src_tokens = merge('source')
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    _, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    target = None
    ntokens = None
    if has_target:
        target = merge('target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
        },
        'target': target,
    }


class MonolingualDataset(Dataset):

    def __init__(self, tokens, sizes, token_block_size, break_mode, pad_idx, eos_idx, next_token_is_target):

        if next_token_is_target:
            self.src = TokenBlockDataset(tokens, token_block_size, sizes, offset=1, break_mode=break_mode)
            self.dst = TokenBlockDataset(tokens, token_block_size, sizes, offset=0, break_mode=break_mode)
        else:
            self.src = TokenBlockDataset(tokens, token_block_size, sizes, offset=0, break_mode=break_mode)
            self.dst = None

        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

    def __getitem__(self, i):
        # subtract 1 for 0-based indexing
        source = self.src[i].long() - 1
        res = {'id': i, 'source': source}
        if self.dst:
            res['target'] = self.dst[i].long() - 1

        return res

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return collate(samples, self.pad_idx, self.eos_idx, self.dst is not None)

    @staticmethod
    def create_dataset(args, splits, is_raw):
        """Loads specified data splits (e.g., test, train or valid) from the
        specified folder and check that files exist."""

        if is_raw:
            raise Exception('raw text single language data sets are currently not supported')

        assert args.sample_break_mode == 'eos' or args.max_target_positions is not None

        path = args.data
        dict = Dictionary.load(os.path.join(path, 'dict.txt'))
        dataset = LanguageDatasets(None, None, dict, dict)

        assert all(os.path.exists(os.path.join(path, '{}.bin'.format(split))) for split in splits)

        for split in splits:
            for k in itertools.count():
                prefix = "{}{}".format(split, k if k > 0 else '')
                split_path = fmt_path(path, '{}', prefix)

                if not IndexedInMemoryDataset.exists(split_path):
                    break

                ds = IndexedInMemoryDataset(split_path)
                tokens = torch.from_numpy(ds.buffer)

                dataset.splits[prefix] = MonolingualDataset(
                    tokens,
                    ds.sizes,
                    args.max_target_positions,
                    args.sample_break_mode,
                    pad_idx=dataset.src_dict.pad(),
                    eos_idx=dataset.src_dict.eos(),
                    next_token_is_target=True,
                )

        return dataset
