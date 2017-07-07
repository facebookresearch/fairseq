import contextlib
import os
import torch
import torch.utils.data
import numpy as np
from indexed_dataset import IndexedDataset
from dictionary import Dictionary


def load(path, src=None, dst=None):
    """Loads the train, valid, and test sets from the specified folder"""

    def find_language_pair(files):
        for filename in files:
            parts = filename.split('.')
            if parts[0] == 'train' and parts[-1] == 'idx':
                return parts[1].split('-')

    def train_file_exists(src, dst):
        filename = 'train.{0}-{1}.{0}.idx'.format(src, dst)
        return os.path.exists(os.path.join(path, filename))

    def fmt_path(fmt, *args):
        return os.path.join(path, fmt.format(*args))

    if src is None and dst is None:
        # find language pair automatically
        src, dst = find_language_pair(os.listdir(path))
        langcode = '{}-{}'.format(src, dst)
    elif train_file_exists(src, dst):
        # check for src-dst langcode
        langcode = '{}-{}'.format(src, dst)
    elif train_file_exists(dst, src):
        # check for dst-src langcode
        langcode = '{}-{}'.format(dst, src)
    else:
        raise ValueError('training file not found for {}-{}'.format(src, dst))

    src_dict = Dictionary.load(fmt_path('dict.{}.txt', src))
    dst_dict = Dictionary.load(fmt_path('dict.{}.txt', dst))
    dataset = LanguageDatasets(src, dst, src_dict, dst_dict)

    for split in ['train', 'valid', 'test']:
        dataset.splits[split] = LanguagePairDataset(
            IndexedDataset(fmt_path('{}.{}.{}', split, langcode, src)),
            IndexedDataset(fmt_path('{}.{}.{}', split, langcode, dst)))

    return dataset


class LanguageDatasets(object):
    def __init__(self, src, dst, src_dict, dst_dict):
        self.src = src
        self.dst = dst
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.splits = {}

    def dataloader(self, split, batch_size=1, num_workers=0, max_tokens=None, seed=None):
        dataset = self.splits[split]
        if split == 'train':
            with numpy_seed(seed):
                batch_sampler = shuffled_batches_by_size(
                    dataset.src, dataset.dst, batch_size, max_tokens)
        else:
            batch_sampler = list(batches_by_size(dataset.src, batch_size, max_tokens))

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=PaddingCollater(self.src_dict.pad()),
            batch_sampler=batch_sampler)


class PaddingCollater(object):
    def __init__(self, padding_value=1):
        self.padding_value = padding_value

    def __call__(self, samples):
        def merge(key, pad_begin):
            return self.merge_with_pad([s[key] for s in samples], pad_begin)

        ntokens = sum(len(s['target']) for s in samples)

        return {
            'id': torch.LongTensor([s['id'].item() for s in samples]),
            'input_tokens': merge('input_tokens', pad_begin=True),
            'input_positions': merge('input_positions', pad_begin=True),
            'target': merge('target', pad_begin=True),
            'src_tokens': merge('src_tokens', pad_begin=False),
            'src_positions': merge('src_positions', pad_begin=False),
            'ntokens': ntokens,
        }

    def merge_with_pad(self, values, pad_begin):
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(self.padding_value)
        for i, v in enumerate(values):
            if pad_begin:
                res[i][size-len(v):].copy_(v)
            else:
                res[i][:len(v)].copy_(v)
        return res


class LanguagePairDataset(object):
    def __init__(self, src, dst, padding_value=1, eos=2):
        self.src = src
        self.dst = dst
        self.padding_value = padding_value
        self.eos = eos

    def __getitem__(self, i):
        src = self.src[i].long() - 1
        target = self.dst[i].long() - 1
        input = target.new(target.size())
        input[0] = self.eos
        input[1:].copy_(target[:-1])

        return {
            'id': i,
            'input_tokens': input,
            'input_positions': self.make_positions(input),
            'target': target,
            'src_tokens': src,
            'src_positions': self.make_positions(src),
        }

    def make_positions(self, x):
        start = self.padding_value + 1
        return torch.arange(start, start + len(x)).type_as(x)

    def __len__(self):
        return len(self.src)


def batches_by_size(dataset, batch_size, max_tokens=None):
    """Returns batches of indices sorted by size. Sequences of different lengths
    are not allowed in the same batch."""
    assert isinstance(dataset, IndexedDataset)
    if max_tokens is None:
        max_tokens = float('Inf')
    sizes = dataset.sizes
    indices = np.argsort(sizes, kind='mergesort')

    batch = []

    def yield_batch(next_idx):
        if len(batch) == 0:
            return False
        if len(batch) == batch_size:
            return True
        if sizes[batch[0]] != sizes[next_idx]:
            return True
        if len(batch) * sizes[next_idx] > max_tokens:
            return True
        return False

    for idx in indices:
        if yield_batch(idx):
            yield batch
            batch = []
        batch.append(idx)

    if len(batch) > 0:
        yield batch


def shuffled_batches_by_size(src, dst, batch_size=1, max_tokens=None):
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
        seq_len = 0

        for idx in indices:
            sample_len = max(src.sizes[idx], dst.sizes[idx])
            if len(batch) > 0 and (len(batch) == batch_size
                                   or seq_len + sample_len > max_tokens):
                yield batch
                batch = []
                seq_len = 0

            batch.append(idx)
            seq_len += sample_len

        if len(batch) > 0:
            yield batch

    batches = list(make_batches())
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
