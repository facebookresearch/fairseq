# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import glob
import math
import numbers
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.utils.data

from fairseq.data.dictionary import Dictionary
from fairseq.data.indexed_dataset import SizedDataset


def has_binary_files(data_dir, splits):
    for split in splits:
        if len(glob.glob(os.path.join(data_dir, '{}*.bin'.format(split)))) == 0:
            return False
    return True


def infer_language_pair(path, splits):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        for split in splits:
            if len(parts) >= 3 and parts[0] == split and parts[-1] == 'idx':
                src, dst = parts[1].split('-')
                break
    return src, dst


def load_dictionaries(path, src_lang, dst_lang):
    """Load dictionaries for a given language pair."""
    src_dict = Dictionary.load(os.path.join(path, 'dict.{}.txt'.format(src_lang)))
    dst_dict = Dictionary.load(os.path.join(path, 'dict.{}.txt'.format(dst_lang)))
    return src_dict, dst_dict


def fmt_path(path, fmt, *args):
    return os.path.join(path, fmt.format(*args))


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
            copy_tensor(v, res[i][size - len(v):])
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
                  ignore_invalid_inputs=False, allow_different_src_lens=False,
                  required_batch_size_multiple=1):
    batch = []
    mult = required_batch_size_multiple

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
    sample_lens = []
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

        sample_lens.append(max(src_size, dst_size))
        sample_len = max(sample_len, sample_lens[-1])
        num_tokens = (len(batch) + 1) * sample_len
        if yield_batch(idx, num_tokens):
            mod8_len = max(mult * (len(batch) // mult), len(batch) % mult)
            yield batch[:mod8_len]
            batch = batch[mod8_len:]
            sample_lens = sample_lens[mod8_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

        batch.append(idx)

    if len(batch) > 0:
        yield batch

    if len(ignored) > 0:
        print("Warning! {} samples are either too short or too long "
              "and will be ignored, first few sample ids={}".format(len(ignored), ignored[:10]))


def batches_by_size(src, dst, max_tokens=None, max_sentences=None,
                    max_positions=(1024, 1024), ignore_invalid_inputs=False,
                    descending=False, required_batch_size_multiple=1, allow_different_src_lens=False):
    """Returns batches of indices sorted by size. Sequences with different
    source lengths are not allowed in the same batch."""
    assert isinstance(src, SizedDataset) and (dst is None or isinstance(dst, SizedDataset))
    if max_tokens is None:
        max_tokens = float('Inf')
    if max_sentences is None:
        max_sentences = float('Inf')
    indices = np.argsort(src.sizes, kind='mergesort')
    if descending:
        indices = np.flip(indices, 0)
    return list(_make_batches(
        src, dst, indices, max_tokens, max_sentences, max_positions,
        ignore_invalid_inputs, allow_different_src_lens=allow_different_src_lens,
        required_batch_size_multiple=required_batch_size_multiple,
    ))


def uneven_batches_by_size(src, dst, max_tokens=None, max_sentences=None,
                           max_positions=(1024, 1024),
                           required_batch_size_multiple=1):
    """Returns batches of indices bucketed by size. Batches may contain
    sequences of different lengths."""
    assert isinstance(src, SizedDataset) and isinstance(dst, SizedDataset)
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
        ignore_invalid_inputs=True, allow_different_src_lens=True,
        required_batch_size_multiple=required_batch_size_multiple,
    ))
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


def get_dummy_batch(ntokens, src_dict, dst_dict, src_len=128, tgt_len=128):
    bsz = int(ntokens / max(src_len, tgt_len))
    bsz = math.ceil(bsz / 8) * 8
    assert src_dict.pad() == dst_dict.pad()
    pad_idx = src_dict.pad()
    src_vocab, dst_vocab = len(src_dict), len(dst_dict)
    dummy_batch = {}
    dummy_batch['id'] = Variable(torch.arange(bsz).long().cuda())
    dummy_batch['ntokens'] = tgt_len * bsz
    dummy_batch['target'] = Variable(torch.Tensor(bsz, tgt_len).uniform_(pad_idx + 1, dst_vocab - 1).long().cuda())
    input = {}
    input['prev_output_tokens'] = Variable(dummy_batch['target'].data.clone())
    input['src_lengths'] = Variable(torch.LongTensor(bsz).fill_(src_len).cuda())
    input['src_tokens'] = Variable(torch.Tensor(bsz, src_len).uniform_(pad_idx + 1, src_vocab - 1).long().cuda())
    dummy_batch['net_input'] = input
    return dummy_batch
