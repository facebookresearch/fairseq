import bisect

import numpy as np
from . import FairseqDataset


class ConcatDataset(FairseqDataset):

    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            l = ratio * len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, sample_ratios=1):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cummulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cummulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cummulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cummulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def sizes(self):
        return np.concatenate([np.tile(ds.sizes, sr) for ds, sr in zip(self.datasets, self.sample_ratios)])

    @property
    def supports_prefetch(self):
        return all([d.supports_prefetch for d in self.datasets])

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cummulative_sizes, self.datasets):
            real_size = len(ds)
            ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to
