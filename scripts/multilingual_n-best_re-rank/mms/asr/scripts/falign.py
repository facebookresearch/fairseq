import os
import tempfile
import re
import librosa
import torch
import json
import numpy as np
import argparse
from tqdm import tqdm
import math

from lib import falign_ext

parser = argparse.ArgumentParser()
parser.add_argument("--txt", type=str)
parser.add_argument("--lid", type=str)
parser.add_argument("--p", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--token_list_dir", type=str, default="/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/dict")
args = parser.parse_args()

mapping = {"cmn":"cmn-script_simplified", "srp":"srp-script_latin", "urd":"urd-script_arabic", "uzb":"uzb-script_latin", "yue":"yue-script_traditional", "aze":"azj-script_latin", "kmr":"kmr-script_latin"}

if __name__ == "__main__":
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    tokens = {}     
    for filename in os.listdir(args.token_list_dir):
        lang = filename.split(".")[0]
        toks = [x.split(" ")[0] for x in open(os.path.join(args.token_list_dir, filename), "r").readlines()]
        # starts after <s> <pad> </s> <unk>
        toks = ['<s>', '<pad>', '</s>', '<unk>'] + toks
        tokens[lang] = toks

    txts = [x.strip() for x in open(args.txt, "r").readlines()]
    lids = [x.strip() for x in open(args.lid, "r").readlines()]
    # probs = [eval(x.strip().replace('-inf', 'float(\'-inf\')')) for x in open(args.p, "r").readlines()]
    probs = [x.strip() for x in open(args.p, "r").readlines()]
    assert len(txts) == len(probs)
    assert len(txts) == len(lids)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # clear it
    with open(args.dst + "/score", "w") as f1:
        pass

    for i, p in tqdm(enumerate(probs)):
        if lids[i] in mapping:
            lang = mapping[lids[i]]
        else:
            lang = lids[i]

        p = eval(p.replace('-inf', 'float(\'-inf\')'))
        emissions = torch.tensor(p, device=device)
        
        chars = txts[i].lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(" ", "|")
        token_sequence = [tokens[lang].index(x) for x in chars if x in tokens[lang]]
        
        try:
            _, alphas, _ = falign_ext.falign(emissions, torch.tensor(token_sequence, device=device).int(), False)
            aligned_alpha = max(alphas[-1]).item()
        except:
            aligned_alpha = math.log(0.000000001)

        print(aligned_alpha)

        with open(args.dst + "/score", "a") as f1:
            f1.write(str(aligned_alpha) + "\n")
            f1.flush()