# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import itertools
import glob
import math
import numbers
import numpy as np
import os
import torch
import torch.utils.data

from fairseq.dictionary import Dictionary
from fairseq.indexed_dataset import IndexedDataset, IndexedInMemoryDataset, IndexedRawTextDataset


def has_binary_files(data_dir, splits):
    for split in splits:
        if len(glob.glob(os.path.join(data_dir, '{}.*-*.*.bin'.format(split)))) < 2:
            return False
    return True


def infer_language_pair(path, splits):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        for split in splits:
            if parts[0] == split and parts[-1] == 'idx':
                src, dst = parts[1].split('-')
                break
    return src, dst


def load_dictionaries(path, src_lang, dst_lang):
    """Load dictionaries for a given language pair."""
    src_dict = Dictionary.load(os.path.join(path, 'dict.{}.txt'.format(src_lang)))
    dst_dict = Dictionary.load(os.path.join(path, 'dict.{}.txt'.format(dst_lang)))
    return src_dict, dst_dict


def load_dataset(path, load_splits, src=None, dst=None):
    """Loads specified data splits (e.g., test, train or valid) from the
    specified folder and check that files exist."""
    if src is None and dst is None:
        # find language pair automatically
        src, dst = infer_language_pair(path, load_splits)
    assert src is not None and dst is not None, 'Source and target languages should be provided'

    src_dict, dst_dict = load_dictionaries(path, src, dst)
    dataset = LanguageDatasets(src, dst, src_dict, dst_dict)

    # Load dataset from binary files
    def all_splits_exist(src, dst, lang):
        for split in load_splits:
            filename = '{0}.{1}-{2}.{3}.idx'.format(split, src, dst, lang)
            if not os.path.exists(os.path.join(path, filename)):
                return False
        return True

    # infer langcode
    if all_splits_exist(src, dst, src):
        langcode = '{}-{}'.format(src, dst)
    elif all_splits_exist(dst, src, src):
        langcode = '{}-{}'.format(dst, src)
    else:
        raise Exception('Dataset cannot be loaded from path: ' + path)

    def fmt_path(fmt, *args):
        return os.path.join(path, fmt.format(*args))

    for split in load_splits:
        for k in itertools.count():
            prefix = "{}{}".format(split, k if k > 0 else '')
            src_path = fmt_path('{}.{}.{}', prefix, langcode, src)
            dst_path = fmt_path('{}.{}.{}', prefix, langcode, dst)

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


def load_raw_text_dataset(path, load_splits, src=None, dst=None):
    """Loads specified data splits (e.g., test, train or valid) from raw text
    files in the specified folder."""
    if src is None and dst is None:
        # find language pair automatically
        src, dst = infer_language_pair(path, load_splits)
    assert src is not None and dst is not None, 'Source and target languages should be provided'

    src_dict, dst_dict = load_dictionaries(path, src, dst)
    dataset = LanguageDatasets(src, dst, src_dict, dst_dict)

    # Load dataset from raw text files
    for split in load_splits:
        src_path = os.path.join(path, '{}.{}'.format(split, src))
        dst_path = os.path.join(path, '{}.{}'.format(split, dst))
        dataset.splits[split] = LanguagePairDataset(
            IndexedRawTextDataset(src_path, src_dict),
            IndexedRawTextDataset(dst_path, dst_dict),
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

    def train_dataloader(self, split, max_tokens=None,
                         max_sentences=None, max_positions=(1024, 1024),
                         seed=None, epoch=1, sample_without_replacement=0,
                         sort_by_source_size=False, shard_id=0, num_shards=1):
        dataset = self.splits[split]
        with numpy_seed(seed):
            batch_sampler = shuffled_batches_by_size(
                dataset.src, dataset.dst, max_tokens=max_tokens,
                max_sentences=max_sentences, epoch=epoch,
                sample=sample_without_replacement, max_positions=max_positions,
                sort_by_source_size=sort_by_source_size)
            batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)
        return torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collater,
            batch_sampler=batch_sampler)

    def eval_dataloader(self, split, num_workers=0, max_tokens=None,
                        max_sentences=None, max_positions=(1024, 1024),
                        skip_invalid_size_inputs_valid_test=False,
                        descending=False, shard_id=0, num_shards=1):
        dataset = self.splits[split]
        batch_sampler = batches_by_size(
            dataset.src, dataset.dst, max_tokens, max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs_valid_test,
            descending=descending)
        batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)
        return torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, collate_fn=dataset.collater,
            batch_sampler=batch_sampler)


class sharded_iterator(object):

    def __init__(self, itr, num_shards, shard_id):
        assert shard_id >= 0 and shard_id < num_shards
        self.itr = itr
        self.num_shards = num_shards
        self.shard_id = shard_id

    def __len__(self):
        return len(self.itr)

    def __iter__(self):
        for i, v in enumerate(self.itr):
            if i % self.num_shards == self.shard_id:
                yield v


class LanguagePairDataset(torch.utils.data.Dataset):

    # padding constants
    LEFT_PAD_SOURCE = True
    LEFT_PAD_TARGET = False

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
        return LanguagePairDataset.collate(samples, self.pad_idx, self.eos_idx, self.dst is not None)

    @staticmethod
    def collate(samples, pad_idx, eos_idx, has_target=True):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False):
            return LanguagePairDataset.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=LanguagePairDataset.LEFT_PAD_SOURCE)
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        prev_output_tokens = None
        target = None
        ntokens = None
        if has_target:
            target = merge('target', left_pad=LanguagePairDataset.LEFT_PAD_TARGET)
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=LanguagePairDataset.LEFT_PAD_TARGET,
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

    @staticmethod
    def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
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


def _valid_size(src_size, dst_size, max_positions):
    if isinstance(max_positions, numbers.Number):
        max_src_positions, max_dst_positions = max_positions, max_positions
    else:
        max_src_positions, max_dst_positions = max_positions
    if src_size < 1 or src_size > max_src_positions:
        return False
    if dst_size is not None and (dst_size < 1 or dst_size > max_dst_positions):
        return False
    return True


def _make_batches(src, dst, indices, max_tokens, max_sentences, max_positions,
                  ignore_invalid_inputs=False, allow_different_src_lens=False):
    batch = []

    def yield_batch(next_idx, num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        if not allow_different_src_lens and \
                (src.sizes[batch[0]] != src.sizes[next_idx]):
            return True
        return False

    sample_len = 0
    ignored = []
    for idx in map(int, indices):
        src_size = src.sizes[idx]
        dst_size = dst.sizes[idx] if dst else src_size
        if not _valid_size(src_size, dst_size, max_positions):
            if ignore_invalid_inputs:
                ignored.append(idx)
                continue
            raise Exception((
                "Sample #{} has size (src={}, dst={}) but max size is {}."
                " Skip this example with --skip-invalid-size-inputs-valid-test"
            ).format(idx, src_size, dst_size, max_positions))

        sample_len = max(sample_len, src_size, dst_size)
        num_tokens = (len(batch) + 1) * sample_len
        while yield_batch(idx, num_tokens):
            mod8_len = max(8 * (len(batch) // 8), len(batch) % 8)
            yield batch[:mod8_len]
            batch = batch[mod8_len:]
            sample_len = max([max(src.sizes[id], dst.sizes[id]) for id in batch]) if len(batch) > 0 else 0
            sample_len = max(sample_len, src_size, dst_size)
            num_tokens = (len(batch) + 1) * sample_len

        batch.append(idx)

    if len(batch) > 0:
        yield batch

    if len(ignored) > 0:
        print("Warning! {} samples are either too short or too long "
              "and will be ignored, first few sample ids={}".format(len(ignored), ignored[:10]))


def batches_by_size(src, dst, max_tokens=None, max_sentences=None,
                    max_positions=(1024, 1024), ignore_invalid_inputs=False,
                    descending=False):
    """Returns batches of indices sorted by size. Sequences with different
    source lengths are not allowed in the same batch."""
    assert isinstance(src, IndexedDataset) and (dst is None or isinstance(dst, IndexedDataset))
    if max_tokens is None:
        max_tokens = float('Inf')
    if max_sentences is None:
        max_sentences = float('Inf')
    indices = np.argsort(src.sizes, kind='mergesort')
    if descending:
        indices = np.flip(indices, 0)
    return list(_make_batches(
        src, dst, indices, max_tokens, max_sentences, max_positions,
        ignore_invalid_inputs, allow_different_src_lens=False))


def shuffled_batches_by_size(src, dst, max_tokens=None, max_sentences=None,
                             epoch=1, sample=0, max_positions=(1024, 1024),
                             sort_by_source_size=False):
    """Returns batches of indices, bucketed by size and then shuffled. Batches
    may contain sequences of different lengths."""
    assert isinstance(src, IndexedDataset) and isinstance(dst, IndexedDataset)
    if max_tokens is None:
        max_tokens = float('Inf')
    if max_sentences is None:
        max_sentences = float('Inf')

    indices = np.random.permutation(len(src))

    # sort by sizes
    indices = indices[np.argsort(dst.sizes[indices], kind='mergesort')]
    indices = indices[np.argsort(src.sizes[indices], kind='mergesort')]

    batches = list(_make_batches(
        src, dst, indices, max_tokens, max_sentences, max_positions,
        ignore_invalid_inputs=True, allow_different_src_lens=True))

    if not sort_by_source_size:
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

    return batches


def mask_batches(batch_sampler, shard_id, num_shards):
    if num_shards == 1:
        return batch_sampler
    res = [
        batch
        for i, batch in enumerate(batch_sampler)
        if i % num_shards == shard_id
    ]
    expected_length = int(math.ceil(len(batch_sampler) / num_shards))
    return res + [[]] * (expected_length - len(res))


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
