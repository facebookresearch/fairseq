#!/usr/bin/env python3
# -*- encoding: utf8 -*-
import argparse
import itertools
import os
import re
import sys 
from pathlib import Path

import whisper
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--wavs", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--use_fallback", type=str, default="true")
parser.add_argument("--model", type=str)
parser.add_argument("--top_k", type=int, default=10)
args = parser.parse_args()

if __name__ == "__main__":
    model = whisper.load_model(args.model)

    print(args)
    
    wavs = [x.strip() for x in open(args.wavs, "r").readlines()]
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    with open(args.dst + "/predictions.txt", "w") as f:
        for wav in tqdm(wavs):
            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(wav)
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            _, probs = model.detect_language(mel)
            result = sorted(probs.items(), key=lambda x:x[1], reverse=True)[:args.top_k]
            f.write(str(result) + "\n")
