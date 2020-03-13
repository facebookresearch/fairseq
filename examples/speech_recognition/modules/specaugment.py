"""SpecAugment Implementation for Pytorch.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
"""

import torch
import torch.nn as nn
import numpy as np
import random

class SpecAugment(nn.Module):
    """
    Class that handles 'SpecAugment'. It stores the parameters and calls the function according to the probabilistic rate.
    """

    def __init__(self, frequency_masking_pars, time_masking_pars,
                 frequency_masking_num, time_masking_num, rate=1.0):
        super().__init__()
        self.frequency_masking_pars = frequency_masking_pars
        self.time_masking_pars = time_masking_pars
        self.frequency_masking_num = frequency_masking_num
        self.time_masking_num = time_masking_num
        self.rate = rate

    def forward(self, batch):
        """
        Performs 'SpecAugment' on a batch

        Args:
            batch (dict): A batch as returned by the 'collater' method

        Returns:
            dict: The same batch with the masking applied by 'SpecAugment'
        """
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
    """
    Spec augmentation Calculation Function.
    'SpecAugment' has 3 steps for audio data augmentation, but only 2 are implemented in this version.
    First step is frequency masking, second step is time masking.

    Args:
        mel_spectrogram (FloatTensor): 2-dimensional Tensor representing a spectrogram
        frequency_masking_para (int): Maximum masking width over frequencies
        time_masking_para (int): Maximum masking width over time
        frequency_masking_num (int): Number of masks on frequencies
        time_masking_num (int): Number of masks on time

    Returns:
        FloatTensor: The masked spectrogram
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
