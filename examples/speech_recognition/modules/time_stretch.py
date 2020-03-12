import torch
import torch.nn as nn
import numpy as np
import random
import copy

class TimeStretch(nn.Module):

    def __init__(self, w, low, high):
        super().__init__()
        if w < 1:
            raise ValueError("w must be greater than 1")
        self.w = w
        self.low = low
        self.high = high

    def forward(self, batch):
        new_batch = copy.deepcopy(batch)
        tokens, lengths = [], []
        for elem, length in zip(batch['net_input']['src_tokens'], batch['net_input']['src_lengths']):
            tokens += [time_stretch_seq(elem[:length, :], self.w, self.low, self.high)]
            lengths += [tokens[-1].size(0)]

        frames = torch.zeros([len(lengths), max(lengths), tokens[0].size(1)], dtype=torch.float32)
        for i, sample in enumerate(tokens):
            frames[i, :sample.size(0), :] = sample

        new_batch['net_input']['src_tokens'] = frames
        new_batch['net_input']['src_lengths'] = torch.Tensor(lengths).long()

        if batch['net_input']['src_tokens'].is_cuda:
            new_batch['net_input']['src_tokens'] = new_batch['net_input']['src_tokens'].cuda()
            new_batch['net_input']['src_lengths'] = new_batch['net_input']['src_lengths'].cuda()

        return new_batch

def time_stretch_seq(mel_spectrogram, w, low=0.8, high=1.25):
    """
    """
    ids = []
    time_len = mel_spectrogram.size(0)
    if time_len < 10 and low < 1.0:
       low = 1.0
    for i in range(int(round(time_len / w))):
        s = random.uniform(low, high) * min(w, time_len-w*i)
        e = min(time_len, w*(i+1))
        r = torch.linspace(w*i, e-1, int(s))
        r = torch.round(r).long()
        if len(ids) == 0:
            ids = r
        else:
            ids = torch.cat((ids, r), dim=0)
    return mel_spectrogram[ids, :]

if __name__ == '__main__':
    unittest.main()
