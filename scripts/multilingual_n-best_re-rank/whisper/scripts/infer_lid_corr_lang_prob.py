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
parser.add_argument("--lids", type=str) #refs
parser.add_argument("--dst", type=str)
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--use_fallback", type=str, default="true")
parser.add_argument("--model", type=str)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument('--mapping', type=str, default="/private/home/yanb/whisper/scripts/lid_mapping.txt")
args = parser.parse_args()

if __name__ == "__main__":
    model = whisper.load_model(args.model)

    print(args)

    if args.mapping is not None:
        # mms_lid_code:whisper_lid_code
        mapping = {x[1]:x[0] for x in [l.strip().split(":", 1) for l in open(args.mapping, "r").readlines()]}
    else:
        mapping = None
    
    wavs = [x.strip() for x in open(args.wavs, "r").readlines()]
    ref_lids = [x.strip() for x in open(args.lids, "r").readlines()]
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    with open(args.dst + "/corr_lang_prob", "w") as f:
        for wav, ref_lid in tqdm(zip(wavs, ref_lids)):
            ref_lid = mapping[ref_lid]

            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(wav)
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            _, probs = model.detect_language(mel)
            if ref_lid in probs:
                val = probs[ref_lid]
            else:
                val = 0
            # corr_lang_probs.append(val)
            f.write(str(val) + "\n")
