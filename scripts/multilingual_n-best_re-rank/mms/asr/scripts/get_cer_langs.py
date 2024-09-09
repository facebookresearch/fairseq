import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy
import re
import subprocess

def get_dict(dict_txt):
    rv = [x.strip().split() for x in open(dict_txt, "r").readlines()]
    return {x[0]:int(x[1]) for x in rv}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--langs', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    html_lines = open(args.langs, "r").readlines()
    langs = []
    for l in html_lines:
        toks = l.split()
        if toks[0] == "<p>" and toks[1] != "Iso":
            langs.append(toks[1])
    
    char_langs = []
    for l in langs:
        chars = get_dict("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/dict/" + l + ".txt")
        if "|" in chars:
            # word based
            continue
        else:
            # char based
            char_langs.append(l)

    with open(args.dst, "w") as f:
        f.writelines([x + "\n" for x in char_langs])