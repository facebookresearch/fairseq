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
parser.add_argument("--lids", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--model", type=str)
parser.add_argument("--mapping", type=str, default="whisper/lid_mapping.txt")
parser.add_argument("--n", type=int, default=10)

args = parser.parse_args()

if __name__ == "__main__":
    model = whisper.load_model(args.model)

    print(args)
    
    wavs = [y for y in [x.strip() for x in open(args.wavs, "r").readlines()] for _ in range(args.n)]
    lids = [x.strip() for x in open(args.lids, "r").readlines()]
    assert len(wavs) == len(lids)

    if args.mapping is not None:
        # mms_lid_code:whisper_lid_code
        mapping = {x[1]:x[0] for x in [l.strip().split(";", 1) for l in open(args.mapping, "r").readlines()]}
    else:
        mapping = None

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    # clear it
    with open(args.dst + "/nbest_asr_hyp", "w") as f1, open(args.dst + "/asr_score", "w") as f2:
        pass
    
    for wav, lang in tqdm(zip(wavs, lids)):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(wav)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        if mapping is not None and lang in mapping.keys():
            lang_code = mapping[lang]
        else:
            lang_code = lang

        # decode the audio
        options = whisper.DecodingOptions(beam_size=args.beam_size, language=lang_code)
        output = whisper.decode(model, mel, options)
        result = output.text
        length = len(output.tokens)
        score = output.avg_logprob * length

        with open(args.dst + "/nbest_asr_hyp", "a") as f1, open(args.dst + "/asr_score", "a") as f2:
            f1.write(result + "\n")
            f2.write(str(score) + "\n")
            f1.flush()
            f2.flush()