# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import contextlib
import itertools
import numpy as np
import os
import torch
import torch.utils.data

from fairseq.dictionary import Dictionary
from fairseq.indexed_dataset import IndexedDataset, IndexedInMemoryDataset


def load_with_check(path, load_splits, src=None, dst=None):
    """Loads the train, valid, and test sets from the specified folder
    and check that training files exist."""

    def find_language_pair(files):
        for filename in files:
            parts = filename.split('.')
            if parts[0] == 'train' and parts[-1] == 'idx':
                return parts[1].split('-')

    def train_file_exists(src, dst):
        filename = 'train.{0}-{1}.{0}.idx'.format(src, dst)
        return os.path.exists(os.path.join(path, filename))

    if src is None and dst is None:
        # find language pair automatically
        src, dst = find_language_pair(os.listdir(path))
    elif train_file_exists(src, dst):
        # check for src-dst langcode
        pass
    elif train_file_exists(dst, src):
        # check for dst-src langcode
        src, dst = dst, src
    else:
        raise ValueError('training file not found for {}-{}'.format(src, dst))

    dataset = load(path, load_splits, src, dst)
    return dataset


def load(path, load_splits, src, dst):
    """Loads specified data splits (e.g. test, train or valid) from the path."""

    langcode = '{}-{}'.format(src, dst)

    def fmt_path(fmt, *args):
        return os.path.join(path, fmt.format(*args))

    src_dict = Dictionary.load(fmt_path('dict.{}.txt', src))
    dst_dict = Dictionary.load(fmt_path('dict.{}.txt', dst))
    dataset = LanguageDatasets(src, dst, src_dict, dst_dict)

    for split in load_splits:
        for k in itertools.count():
            prefix = "{}{}".format(split, k if k > 0 else '')
            src_path = fmt_path('{}.{}.{}', prefix, langcode, src)

            if not IndexedInMemoryDataset.exists(src_path):
                break

            dataset.splits[prefix] = LanguagePairDataset(
                IndexedInMemoryDataset(src_path),
                IndexedInMemoryDataset(fmt_path('{}.{}.{}', prefix, langcode, dst)),
                pad_idx=dataset.src_dict.pad(),
                eos_idx=dataset.src_dict.eos(),
            )

    return dataset


class LanguageDatasets(object):
    def __init__(self, src, dst, src_dict, dst_dict):
        self.src = src
        self.dst = dst
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.splits = {}

        assert self.src_dict.pad() == self.dst_dict.pad()
        assert self.src_dict.eos() == self.dst_dict.eos()
        assert self.src_dict.unk() == self.dst_dict.unk()

    def dataloader(self, split, batch_size=1, num_workers=0,
                   max_tokens=None, seed=None, epoch=1,
                   sample_without_replacement=0, max_positions=1024,
                   skip_invalid_size_inputs_valid_test=False):
        dataset = self.splits[split]
        if split.startswith('train'):
            with numpy_seed(seed):
                batch_sampler = shuffled_batches_by_size(
                    dataset.src, dataset.dst,
                    max_tokens=max_tokens, epoch=epoch,
                    sample=sample_without_replacement,
                    max_positions=max_positions)
        elif split.startswith('valid'):
            batch_sampler = list(batches_by_size(dataset.src, batch_size, max_tokens, dst=dataset.dst,
                                                 max_positions=max_positions,
                                                 ignore_invalid_inputs=skip_invalid_size_inputs_valid_test))
        else:
            batch_sampler = list(batches_by_size(dataset.src, batch_size, max_tokens, max_positions=max_positions,
                                                 ignore_invalid_inputs=skip_invalid_size_inputs_valid_test))

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
        )


def skip_group_enumerator(it, ngpus, offset=0):
    res = []
    idx = 0
    for i, sample in enumerate(it):
        if i < offset:
            continue
        res.append(sample)
        if len(res) >= ngpus:
            yield (i, res)
            res = []
            idx = i + 1
    if len(res) > 0:
        yield (idx, res)


class LanguagePairDataset(object):
    def __init__(self, src, dst, pad_idx, eos_idx):
        self.src = src
        self.dst = dst
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

    def __getitem__(self, i):
        # subtract 1 for 0-based indexing
        source = self.src[i].long() - 1
        target = self.dst[i].long() - 1
        return {
            'id': i,
            'source': source,
            'target': target,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return LanguagePairDataset.collate(samples, self.pad_idx, self.eos_idx)

    @staticmethod
    def collate(samples, pad_idx, eos_idx):

        def merge(key, left_pad, move_eos_to_beginning=False):
            return LanguagePairDataset.collate_tokens(
                [s[key] for s in samples], pad_idx, eos_idx, left_pad, move_eos_to_beginning)

        def merge_positions(key, left_pad):
            return LanguagePairDataset.collate_positions([s[key] for s in samples], pad_idx, left_pad)

        ntokens = sum(len(s['target']) for s in samples)
        return {
            'id': torch.LongTensor([s['id'].item() for s in samples]),
            'input_tokens': merge('target', left_pad=True, move_eos_to_beginning=True),
            'input_positions': merge_positions('target', left_pad=True),
            'target': merge('target', left_pad=True),
            'src_tokens': merge('source', left_pad=False),
            'src_positions': merge_positions('source', left_pad=False),
            'ntokens': ntokens,
        }

    @staticmethod
    def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning):
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            if left_pad:
                copy_tensor(v, res[i][size-len(v):])
            else:
                copy_tensor(v, res[i][:len(v)])
        return res

    @staticmethod
    def collate_positions(values, pad_idx, left_pad):
        start = pad_idx + 1
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)
        for i, v in enumerate(values):
            if left_pad:
                torch.arange(start, start + len(v), out=res[i][size-len(v):])
            else:
                torch.arange(start, start + len(v), out=res[i][:len(v)])
        return res


def batches_by_size(src, batch_size=None, max_tokens=None, dst=None,
                    max_positions=1024, ignore_invalid_inputs=False):
    """Returns batches of indices sorted by size. Sequences of different lengths
    are not allowed in the same batch."""
    assert isinstance(src, IndexedDataset)
    assert dst is None or isinstance(dst, IndexedDataset)
    if max_tokens is None:
        max_tokens = float('Inf')
    sizes = src.sizes
    indices = np.argsort(sizes, kind='mergesort')
    if dst is not None:
        sizes = np.maximum(sizes, dst.sizes)

    batch = []

    def yield_batch(next_idx, num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == batch_size:
            return True
        if sizes[batch[0]] != sizes[next_idx]:
            return True
        if num_tokens >= max_tokens:
            return True
        return False

    cur_max_size = 0
    ignored = []
    for idx in indices:
        # - 2 here stems from make_positions() where we offset positions
        # by padding_value + 1
        if src.sizes[idx] < 2 or \
                (False if dst is None else dst.sizes[idx] < 2) or \
                sizes[idx] > max_positions - 2:
            if ignore_invalid_inputs:
                ignored.append(idx)
                continue

            raise Exception("Unable to handle input id {} of "
                            "size {} / {}.".format(idx, src.sizes[idx],
                                                   "none" if dst is None else dst.sizes[idx]))

        if yield_batch(idx, cur_max_size * (len(batch) + 1)):
            yield batch
            batch = []
            cur_max_size = 0
        batch.append(idx)
        cur_max_size = max(cur_max_size, sizes[idx])

    if len(ignored) > 0:
        print("Warning! {} samples are either too short or too long "
              "and will be ignored, sample ids={}".format(len(ignored), ignored))

    if len(batch) > 0:
        yield batch


def shuffled_batches_by_size(src, dst, max_tokens=None, epoch=1, sample=0, max_positions=1024):
    """Returns batches of indices, bucketed by size and then shuffled. Batches
    may contain sequences of different lengths."""
    assert isinstance(src, IndexedDataset) and isinstance(dst, IndexedDataset)
    if max_tokens is None:
        max_tokens = float('Inf')

    indices = np.random.permutation(len(src))

    # sort by sizes
    indices = indices[np.argsort(dst.sizes[indices], kind='mergesort')]
    indices = indices[np.argsort(src.sizes[indices], kind='mergesort')]

    def make_batches():
        batch = []
        sample_len = 0
        ignored = []
        for idx in indices:
            # - 2 here stems from make_positions() where we offset positions
            # by padding_value + 1
            if src.sizes[idx] < 2 or dst.sizes[idx] < 2 or \
                            src.sizes[idx] > max_positions - 2 or \
                            dst.sizes[idx] > max_positions - 2:
                ignored.append(idx)
                continue
            sample_len = max(sample_len, src.sizes[idx], dst.sizes[idx])
            if len(batch) > 0 and (len(batch) + 1) * sample_len > max_tokens:
                yield batch
                batch = []
                sample_len = max(src.sizes[idx], dst.sizes[idx])

            batch.append(idx)

        if len(batch) > 0:
            yield batch

        if len(ignored) > 0:
            print("Warning! {} samples are either too short or too long "
                  "and will be ignored, sample ids={}".format(len(ignored), ignored))

    batches = list(make_batches())
    np.random.shuffle(batches)

    if sample:
        offset = (epoch - 1) * sample
        while offset > len(batches):
            np.random.shuffle(batches)
            offset -= len(batches)

        result = batches[offset:(offset + sample)]
        while len(result) < sample:
            np.random.shuffle(batches)
            result += batches[:(sample - len(result))]

        assert len(result) == sample, \
            "batch length is not correct {}".format(len(result))

        batches = result
    else:
        for i in range(epoch - 1):
            np.random.shuffle(batches)

    return batches


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
