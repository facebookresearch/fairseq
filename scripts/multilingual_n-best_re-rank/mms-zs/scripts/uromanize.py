import os
import tempfile
import re
import librosa
import torch
import json
import numpy as np
import argparse
from tqdm import tqdm

uroman_dir = "uroman"
assert os.path.exists(uroman_dir)
UROMAN_PL = os.path.join(uroman_dir, "bin", "uroman.pl")

parser = argparse.ArgumentParser()
parser.add_argument("--txt", type=str)
parser.add_argument("--lid", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

def norm_uroman(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()

def uromanize(words):
    iso = "xxx"
    with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
        with open(tf.name, "w") as f:
            f.write("\n".join(words))
        cmd = f"perl " + UROMAN_PL
        cmd += f" -l {iso} "
        cmd += f" < {tf.name} > {tf2.name}"
        os.system(cmd)
        lexicon = {}
        with open(tf2.name) as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                line = re.sub(r"\s+", "", norm_uroman(line)).strip()
                lexicon[words[idx]] = " ".join(line) + " |"
    return lexicon

def convert_sent(txt, char_lang=False):
    if char_lang:
        words = txt
    else:
        words = txt.split(" ")
    lexicon = uromanize(words)
    pron = []
    pron_no_sp = []
    for w in words:
        if w in lexicon:
            pron.append(lexicon[w])
            pron_no_sp.append(lexicon[w].replace(" |", ""))

    return " ".join(pron), " ".join(pron_no_sp)

if __name__ == "__main__":
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    txts = [x.strip() for x in open(args.txt, "r").readlines()]
    langs = [x.strip() for x in open(args.lid, "r").readlines()]
    assert len(txts) == len(langs)

    cer_langs = [x.strip() for x in open("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/cer_langs.txt", "r").readlines()]

    with open(args.dst + "/pron", "w", buffering=1) as f1, open(args.dst + "/pron_no_sp", "w", buffering=1) as f2:
        for t, l in tqdm(zip(txts,langs), total=len(txts)):
            pron, pron_no_sp = convert_sent(t, l in cer_langs)
            f1.write(pron + "\n")
            f2.write(pron_no_sp + "\n")