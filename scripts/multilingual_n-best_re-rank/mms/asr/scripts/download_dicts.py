import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy
import re
import subprocess

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
    
    for l in langs:
        url = "https://dl.fbaipublicfiles.com/mms/asr/dict/mms1b_all/" + l + ".txt"
        dst = args.dst + "/" + l + ".txt"
        subprocess.run(["wget", url, "-O", dst], check=True)
        