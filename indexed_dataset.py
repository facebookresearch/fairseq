import torch
import torch.utils.data
import struct
import numpy as np


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


class IndexedDataset(object):
    """Loader for TorchNet IndexedDataset"""

    dtypes = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float,
        7: np.double,
    }

    def __init__(self, path):
        with open(path + '.idx', 'rb') as f:
            magic = f.read(8)
            assert magic == b'TNTIDX\x00\x00'
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = self.dtypes[code]
            self.size, self.s = struct.unpack('<QQ', f.read(16))
            self.dim_offsets = read_longs(f, self.size + 1)
            self.data_offsets = read_longs(f, self.size + 1)
            self.sizes = read_longs(f, self.s)
        self.data_file = open(path + '.bin', 'rb', buffering=0)

    def __del__(self):
        self.data_file.close()

    def __getitem__(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        return torch.from_numpy(a)

    def __len__(self):
        return self.size
