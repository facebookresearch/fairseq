import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--wavs', type=str)
    parser.add_argument('--lid_preds', type=str)
    parser.add_argument('--dst', type=str, default=None)
    args = parser.parse_args()


    wavs = [x.strip() for x in open(args.wavs, "r").readlines()]
    lid_preds = [eval(x) for x in open(args.lid_preds, "r").readlines()]

    assert len(wavs) == len(lid_preds)

    topk_wavs = []
    topk_langs = []

    for w, p in zip(wavs, lid_preds):
        if p == "n/a":
            continue
        
        assert len(p) == len(lid_preds[0])

        for l, _ in p:
            topk_wavs.append(w)
            topk_langs.append(l)

    if not os.path.exists(args.dst):
            os.makedirs(args.dst)

    with open(args.dst + "/wav.txt", "w") as f:
        f.writelines([x+"\n" for x in topk_wavs])

    with open(args.dst + "/lid.txt", "w") as f:
        f.writelines([x+"\n" for x in topk_langs])