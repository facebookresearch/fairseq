#!/usr/bin/env python3
# -*- encoding: utf8 -*-
import fasttext
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--txt", type=str)
parser.add_argument("--dst", type=str)
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

    pretrained_lang_model = "lid218e.bin"
    model = fasttext.load_model(pretrained_lang_model)

    # preds = []

    txts = [x.strip() for x in open(args.txt, "r").readlines()]

    with open(args.dst + "/predictions.txt", "w") as f:
        for t in tqdm(txts):
            predictions = model.predict(t, k=218)    # max 218
            predictions = [(fix_code(x), y) for x, y in zip(predictions[0], predictions[1])]
            # preds.append(preds)
            f.write(str(predictions) + "\n")

    # with open(args.dst + "/predictions.txt", "w") as f:
    #     f.writelines([str(x) + "\n" for x in preds])