import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    lines = [x.strip() for x in open(args.src + "/text", "r").readlines()]
    wavs = [x.strip() for x in open(args.src + "/wav.scp", "r").readlines()]

    labels = []
    texts = []
    wavs_new = []
    for l, w in zip(lines, wavs):
        try:
            uttid, text = l.split(" ", 1)
            uttid2, wav = w.split(" ", 1)
            assert uttid == uttid2
        except:
            continue
        label = uttid.split("_")[1]
        labels.append(label)
        texts.append(text)
        wavs_new.append(wav)
    
    with open(args.dst + "/test.label", "w") as f1, \
        open(args.dst + "/test.ltr", "w") as f2, \
        open(args.dst + "/test.wrd", "w") as f3, \
        open(args.dst + "/wav.txt", "w") as f4, \
        open(args.dst + "/test.tsv", "w") as f5:
        f5.write("/\n")
        for l, t, w in zip(labels, texts, wavs_new):
            f1.write(l+"\n")
            f2.write(" ".join(t.replace(" ", "|")) + "\n")
            f3.write(t + "\n")
            f4.write(w + "\n")
            # import pdb;pdb.set_trace()
            samples = sf.SoundFile(w).frames
            f5.write(w + "\t" + str(samples) + "\n")