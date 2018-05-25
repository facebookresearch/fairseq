# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
import torch
import torch.utils

from fairseq.data import LanguageDatasets
from fairseq.data.consts import LEFT_PAD_TARGET, LEFT_PAD_SOURCE
from fairseq.data.data_utils import fmt_path, load_dictionaries, collate_tokens
from fairseq.data.indexed_dataset import IndexedInMemoryDataset, IndexedRawTextDataset


def collate(samples, pad_idx, eos_idx, has_target):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=LEFT_PAD_SOURCE)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    ntokens = None
    if has_target:
        target = merge('target', left_pad=LEFT_PAD_TARGET)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=LEFT_PAD_TARGET,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'prev_output_tokens': prev_output_tokens,
        },
        'target': target,
    }


class LanguagePairDataset(torch.utils.data.Dataset):

    def __init__(self, src, dst, pad_idx, eos_idx):
        self.src = src
        self.dst = dst
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
        src, dst = args.source_lang, args.target_lang
        assert src is not None and dst is not None, 'Source and target languages should be provided'

        src_dict, dst_dict = load_dictionaries(args.data, src, dst)
        dataset = LanguageDatasets(src, dst, src_dict, dst_dict)

        def create_raw_dataset():
            """Loads specified data splits (e.g., test, train or valid) from raw text
                    files in the specified folder."""

            # Load dataset from raw text files
            for split in splits:
                src_path = os.path.join(args.data, '{}.{}'.format(split, src))
                dst_path = os.path.join(args.data, '{}.{}'.format(split, dst))
                dataset.splits[split] = LanguagePairDataset(
                    IndexedRawTextDataset(src_path, src_dict),
                    IndexedRawTextDataset(dst_path, dst_dict),
                    pad_idx=dataset.src_dict.pad(),
                    eos_idx=dataset.src_dict.eos(),
                )
            return dataset

        def create_binary_dataset():
            """Loads specified data splits (e.g., test, train or valid) from the
            specified folder and check that files exist."""

            # Load dataset from binary files
            def all_splits_exist(src, dst, lang):
                for split in splits:
                    filename = '{0}.{1}-{2}.{3}.idx'.format(split, src, dst, lang)
                    if not os.path.exists(os.path.join(args.data, filename)):
                        return False
                return True

            # infer langcode
            if all_splits_exist(src, dst, src):
                langcode = '{}-{}'.format(src, dst)
            elif all_splits_exist(dst, src, src):
                langcode = '{}-{}'.format(dst, src)
            else:
                raise Exception('Dataset cannot be loaded from path: ' + args.data)

            for split in splits:
                for k in itertools.count():
                    prefix = "{}{}".format(split, k if k > 0 else '')
                    src_path = fmt_path(args.data, '{}.{}.{}', prefix, langcode, src)
                    dst_path = fmt_path(args.data, '{}.{}.{}', prefix, langcode, dst)

                    if not IndexedInMemoryDataset.exists(src_path):
                        break

                    target_dataset = None
                    if IndexedInMemoryDataset.exists(dst_path):
                        target_dataset = IndexedInMemoryDataset(dst_path)

                    dataset.splits[prefix] = LanguagePairDataset(
                        IndexedInMemoryDataset(src_path),
                        target_dataset,
                        pad_idx=dataset.src_dict.pad(),
                        eos_idx=dataset.src_dict.eos(),
                    )

            return dataset

        return create_raw_dataset() if is_raw else create_binary_dataset()
