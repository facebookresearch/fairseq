import bisect

from . import FairseqDataset


class ConcatDataset(FairseqDataset):

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cummulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cummulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cummulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cummulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def supports_prefetch(self):
        return all([d.supports_prefetch for d in self.datasets])

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cummulative_sizes, self.datasets):
            ds.prefetch([i - frm for i in indices if frm <= i < to])
            frm = to
