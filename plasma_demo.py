import argparse
from fairseq.data.plasma_utils import PlasmaView, start_plasma_store
from torch.utils.data import DataLoader, Dataset
import fire
import numpy as np


import torch

class PlasmaDataset(Dataset):

    def __init__(self, array, split_path):
        super().__init__()
        self.pv = PlasmaView(array, split_path, 0)

    def __getitem__(self, item):
        return torch.tensor(self.pv.array[item])

    def __len__(self):
        return len(self.pv)


def train(num_workers):
    dtrain = np.random.rand(1000, 10)
    dval = np.random.rand(100, 10)
    train_ds = PlasmaDataset(dtrain, 'train')
    val_ds = PlasmaDataset(dval, 'val')
    train_dl = DataLoader(train_ds, batch_size=4, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=4, num_workers=num_workers)
    for batch in train_dl:
        lens = len(batch)
    for batch in val_dl:
        lens = len(batch)
    print('DONE')


def main():
    server = start_plasma_store()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, type=int)
    args = parser.parse_args()
    #parser.add_argument("--dimension", type=int, default=1024, help="Size of each key")
    train(args.num_workers)
    server.kill()


if __name__ == '__main__':
    main()
