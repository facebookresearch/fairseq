import collections
import io
import json
import librosa
import numpy as np
import soundfile as sf
import time
import torch
from scipy.io.wavfile import read
from .text import SOS_TOK, EOS_TOK


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path, sr=None):
    data, sr = librosa.load(full_path, sr=sr)
    data = np.clip(data, -1, 1)  # potentially out of [-1, 1] due to resampling
    data = data * 32768.0  # match values loaded by scipy
    return torch.FloatTensor(data.astype(np.float32)), sr


def read_binary_audio(bin_data, tar_sr=None):
    """
    read binary audio (`bytes` or `uint8` `numpy.ndarray`) to `float32`
    `numpy.ndarray`

    RETURNS:
        data (np.ndarray) : audio of shape (n,) or (2, n)
        tar_sr (int) : sample rate
    """
    data, ori_sr = sf.read(io.BytesIO(bin_data), dtype='float32')
    data = data.T
    if (tar_sr is not None) and (ori_sr != tar_sr):
        data = librosa.resample(data, ori_sr, tar_sr)
    else:
        tar_sr = ori_sr
    data = np.clip(data, -1, 1)
    data = data * 32768.0
    return torch.FloatTensor(data.astype(np.float32)), tar_sr


def load_filepaths_and_text(filename):
    with open(filename, encoding='utf-8') as f:
        data = [json.loads(line.rstrip()) for line in f]
    return data


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def load_code_dict(path, add_sos=False, add_eos=False):
    if not path:
        return {}

    with open(path, 'r') as f:
        codes = ['_'] + [line.rstrip() for line in f]  # '_' for pad
    code_dict = {c: i for i, c in enumerate(codes)}

    if add_sos:
        code_dict[SOS_TOK] = len(code_dict)
    if add_eos:
        code_dict[EOS_TOK] = len(code_dict)
    assert(set(code_dict.values()) == set(range(len(code_dict))))

    return code_dict


def load_obs_label_dict(path):
    if not path:
        return {}
    with open(path, 'r') as f:
        obs_labels = [line.rstrip() for line in f]
    return {c: i for i, c in enumerate(obs_labels)}


# A simple timer class inspired from `tnt.TimeMeter`
class CudaTimer:
    def __init__(self, keys):
        self.keys = keys
        self.reset()

    def start(self, key):
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        self.start_events[key].append(s)
        return self

    def stop(self, key):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.end_events[key].append(e)
        return self

    def reset(self):
        self.start_events = collections.defaultdict(list)
        self.end_events = collections.defaultdict(list)
        self.running_times = collections.defaultdict(float)
        self.n = collections.defaultdict(int)
        return self

    def value(self):
        self._synchronize()
        return {k: self.running_times[k] / self.n[k] for k in self.keys}

    def _synchronize(self):
        torch.cuda.synchronize()
        for k in self.keys:
            starts = self.start_events[k]
            ends = self.end_events[k]
            if len(starts) == 0:
                raise ValueError("Trying to divide by zero in TimeMeter")
            if len(ends) != len(starts):
                raise ValueError("Call stop before checking value!")
            time = 0
            for start, end in zip(starts, ends):
                time += start.elapsed_time(end)
            self.running_times[k] += time * 1e-3
            self.n[k] += len(starts)
        self.start_events = collections.defaultdict(list)
        self.end_events = collections.defaultdict(list)


# Used to measure the time taken for multiple events
class Timer:
    def __init__(self, keys):
        self.keys = keys
        self.n = {}
        self.running_time = {}
        self.total_time = {}
        self.reset()

    def start(self, key):
        self.running_time[key] = time.time()
        return self

    def stop(self, key):
        self.total_time[key] = time.time() - self.running_time[key]
        self.n[key] += 1
        self.running_time[key] = None
        return self

    def reset(self):
        for k in self.keys:
            self.total_time[k] = 0
            self.running_time[k] = None
            self.n[k] = 0
        return self

    def value(self):
        vals = {}
        for k in self.keys:
            if self.n[k] == 0:
                raise ValueError("Trying to divide by zero in TimeMeter")
            else:
                vals[k] = self.total_time[k] / self.n[k]
        return vals

