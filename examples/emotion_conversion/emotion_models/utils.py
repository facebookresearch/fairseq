import torch


class Stat:
    def __init__(self, keep_raw=False):
        self.x = 0.0
        self.x2 = 0.0
        self.z = 0.0  # z = logx
        self.z2 = 0.0
        self.n = 0.0
        self.u = 0.0
        self.keep_raw = keep_raw
        self.raw = []

    def update(self, new_x):
        new_z = new_x.log()

        self.x += new_x.sum()
        self.x2 += (new_x**2).sum()
        self.z += new_z.sum()
        self.z2 += (new_z**2).sum()
        self.n += len(new_x)
        self.u += 1

        if self.keep_raw:
            self.raw.append(new_x)

    @property
    def mean(self):
        return self.x / self.n

    @property
    def std(self):
        return (self.x2 / self.n - self.mean**2) ** 0.5

    @property
    def mean_log(self):
        return self.z / self.n

    @property
    def std_log(self):
        return (self.z2 / self.n - self.mean_log**2) ** 0.5

    @property
    def n_frms(self):
        return self.n

    @property
    def n_utts(self):
        return self.u

    @property
    def raw_data(self):
        assert self.keep_raw, "does not support storing raw data!"
        return torch.cat(self.raw)


class F0Stat(Stat):
    def update(self, new_x):
        # assume unvoiced frames are 0 and consider only voiced frames
        if new_x is not None:
            super().update(new_x[new_x != 0])


class Accuracy:
    def __init__(self):
        self.y, self.yhat = [], []

    def update(self, yhat, y):
        self.yhat.append(yhat)
        self.y.append(y)

    def acc(self, tol):
        yhat = torch.cat(self.yhat)
        y = torch.cat(self.y)
        acc = torch.abs(yhat - y) <= tol
        acc = acc.float().mean().item()
        return acc
