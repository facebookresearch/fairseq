"""SpecAugment Implementation for Pytorch.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
"""

import torch
import torch.nn as nn
import numpy as np
import random


class SpecAugment(nn.Module):

    def __init__(self, frequency_masking_pars, time_masking_pars,
                 frequency_masking_num, time_masking_num, rate=1.0):
        super().__init__()
        self.frequency_masking_pars = frequency_masking_pars
        self.time_masking_pars = time_masking_pars
        self.frequency_masking_num = frequency_masking_num
        self.time_masking_num = time_masking_num
        self.rate = rate

    def forward(self, batch):
        new_spectrograms = []
        x = batch['net_input']['src_tokens']
        for spectrogram in x:
            if random.random() < self.rate:
                sample = specaugment(spectrogram, self.frequency_masking_pars,
                                    self.time_masking_pars, self.frequency_masking_num, self.time_masking_num,
                                     )
            else:
                sample = spectrogram
            new_spectrograms += [sample]

        new_spectrograms = torch.stack(new_spectrograms)
        batch['net_input']['src_tokens'] = new_spectrograms
        return batch


def specaugment(mel_spectrogram, frequency_masking_para=27,
                 time_masking_para=100, frequency_masking_num=1, time_masking_num=1):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    tau, v = mel_spectrogram.size()

    # Step 1 : Frequency masking (masks can overlap)
    for i in range(frequency_masking_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        mel_spectrogram[:, f0:f0 + f] = 0

    # Step 2 : Time masking (masks can overlap)
    for i in range(time_masking_num):
        t = np.random.uniform(low=1.0, high=min(time_masking_para, tau))
        t = int(t)
        t0 = random.randint(0, tau - t)
        mel_spectrogram[t0:t0 + t, :] = 0

    return mel_spectrogram
