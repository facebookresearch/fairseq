#!/usr/bin/env python3
# -*- encoding: utf8 -*-
import argparse
import itertools
import os
import re
import sys 
from pathlib import Path
import math

import whisper
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--wavs", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--mapping", type=str, default="whisper/lid_mapping.txt")
args = parser.parse_args()

if __name__ == "__main__":
    model = whisper.load_model(args.model)

    print(args)
    
    wavs = [x.strip() for x in open(args.wavs, "r").readlines()]
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    if args.mapping is not None:
        #whisper_lid_code:mms_lid_code
        mapping = {x[0]:x[1] for x in [l.strip().split(";", 1) for l in open(args.mapping, "r").readlines()]}
    else:
        mapping = None

    with open(args.dst + "/predictions", "w") as f:
        for wav in tqdm(wavs):
            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(wav)
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            _, probs = model.detect_language(mel)
            result = sorted(probs.items(), key=lambda x:x[1], reverse=True)[:args.n]
            f.write(str(result) + "\n")

    lid_preds = [eval(x) for x in open(args.dst + "/predictions", "r").readlines()]
    lids = []
    scores = []
    for p in lid_preds:
        assert len(p) == len(lid_preds[0])
        for l, s in p:
            if args.mapping is not None:
                lids.append(mapping[l])
            else:
                lids.append(l)
            scores.append(math.log(s))
    with open(args.dst + "/nbest_lid", "w") as f:
        f.writelines([x+"\n" for x in lids])
    with open(args.dst + "/slid_score", "w") as f:
        f.writelines([str(x)+"\n" for x in scores])