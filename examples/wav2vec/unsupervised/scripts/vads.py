#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

sys.path.append("/path/to/rVADfast_py_2.0")
import speechproc
from copy import deepcopy
from scipy.signal import lfilter

import numpy as np
from tqdm import tqdm
import soundfile as sf
import os.path as osp


def rvad(path):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    opts = 1

    data, fs = sf.read(path)
    assert fs == 16_000, "sample rate must be 16khz"
    ft, flen, fsh10, nfr10 = speechproc.sflux(data, fs, winlen, ovrlen, nftt)

    # --spectral flatness --
    pv01 = np.zeros(ft.shape[0])
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)

    pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata = lfilter(b, a, data, axis=0)

    # --pass 1--
    noise_samp, noise_seg, n_noise_samp = speechproc.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk
    )

    # sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0

    vad_seg = speechproc.snre_vad(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres
    )
    return vad_seg, data


def main():
    stride = 160
    lines = sys.stdin.readlines()
    root = lines[0].rstrip()
    for fpath in tqdm(lines[1:]):
        path = osp.join(root, fpath.split()[0])
        vads, wav = rvad(path)

        start = None
        vad_segs = []
        for i, v in enumerate(vads):
            if start is None and v == 1:
                start = i * stride
            elif start is not None and v == 0:
                vad_segs.append((start, i * stride))
                start = None
        if start is not None:
            vad_segs.append((start, len(wav)))

        print(" ".join(f"{v[0]}:{v[1]}" for v in vad_segs))


if __name__ == "__main__":
    main()
