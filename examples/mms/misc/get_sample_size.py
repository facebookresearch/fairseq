#!/usr/bin/env python -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usage:
    $ python misc/get_sample_size.py <input_file> > <output_file>
    
    <input_file> contains list of wav files
    $ cat <input_file>
      /path/to/audio_1.wav
      /path/to/audio_2.wav

    <output_file> contains list of wav files paired with their number of samples
    $ cat <output_file>
      /path/to/audio_1.wav    180000
      /path/to/audio_2.wav    120000
"""
import sys
import soundfile as sf

if __name__ == "__main__":
    files = sys.argv[1]
    with open(files) as fr:
        for fi in fr:
            fi = fi.strip()
            print(f'{fi}\t{sf.SoundFile(fi).frames}')
