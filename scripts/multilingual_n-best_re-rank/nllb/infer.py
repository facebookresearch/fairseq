#!/usr/bin/env python3
# -*- encoding: utf8 -*-
import fasttext
from tqdm import tqdm
import argparse
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument("--txt", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--model", type=str)
parser.add_argument('--lid', type=str)
args = parser.parse_args()

mapping = {"arb":"ara", "azj":"aze", "pes":"fas", "fuv":"ful", "lvs":"lav", "khk":"mon", "zsm":"zlm", "gaz":"orm", "pbt":"pus", "uzn":"uzb", "zho":"cmn"}

def fix_code(x):
    code = x.split("_")[-2]
    if code in mapping:
        code = mapping[code]
    return code

if __name__ == "__main__":
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    pretrained_lang_model = args.model
    model = fasttext.load_model(pretrained_lang_model)

    txts = [x.strip() for x in open(args.txt, "r").readlines()]
    lids = [x.strip() for x in open(args.lid, "r").readlines()]
    assert len(txts) == len(lids)

    with open(args.dst + "/wlid_score", "w") as f:
        for t,l in tqdm(zip(txts, lids)):
            predictions = model.predict(t, k=218)    # max 218
            predictions = [(fix_code(x), y) for x, y in zip(predictions[0], predictions[1])]

            try:
                pred_langs = [x[0] for x in predictions]
                idx = pred_langs.index(l)
                score = math.log(predictions[idx][-1])
            except:
                score = -1000
            f.write(str(score) + "\n")