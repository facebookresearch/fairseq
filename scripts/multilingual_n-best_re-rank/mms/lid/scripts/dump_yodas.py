import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re
import pandas as pd
import random
import pycountry
import ast
import soundfile as sf

# https://huggingface.co/datasets/espnet/long-yodas-segmented

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--dir', type=str)
    parser.add_argument('--dump', type=str)
    args = parser.parse_args()

    langs = [args.dir + d for d in os.listdir(args.dir)]

    wav = []
    label = []
    text = []
    
    for l in langs:
        text_lines = open(l + "/text.test", "r").readlines()
        for t in text_lines:
            t_data = eval(t)
            wav.append(t_data["wav_path"])
            label.append(t_data["language"])
            text.append(t_data["text"].strip())

    with open(args.dump + "/wav.txt", "w") as f1, \
        open(args.dump + "/test.tsv", "w") as f2, \
        open(args.dump + "/test.label", "w") as f3, \
        open(args.dump + "/test.wrd", "w") as f4, \
        open(args.dump + "/test.ltr", "w") as f5:
        f2.write("/\n")
        for w, l, t in zip(wav, label, text):
            samples = sf.SoundFile(w).frames
            if samples == 0:
                continue
            if t == "":
                continue
            lang = pycountry.languages.get(alpha_2=l).alpha_3
            f1.write(w+"\n")
            f2.write(w+"\t"+str(samples)+"\n")
            f3.write(lang+"\n")
            f4.write(t+"\n")
            f5.write(" ".join(t.replace(" ", "|")) + "\n")