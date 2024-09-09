import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--tsv', type=str)
    parser.add_argument('--lid', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument("--mapping", type=str, default="scripts/lid_mapping.txt")
    args = parser.parse_args()

    mapping = {x[1]:x[0] for x in [l.strip().split(":", 1) for l in open(args.mapping, "r").readlines()]}

    tsv = [x.strip() for x in open(args.tsv, "r").readlines()]
    root = tsv[0]
    wavs = [root + x.split("\t")[0] for x in tsv[1:]]
    lid = [x.strip() for x in open(args.lid, "r").readlines()]

    assert len(wavs) == len(lid)

    new_wavs = []
    new_lid = []

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    for w, l in zip(wavs, lid):
        if l in mapping:
            new_wavs.append(w)
            new_lid.append(l)

    with open(args.dst + "/wav.txt", "w") as f1, open(args.dst + "/lid.txt", "w") as f2:
        f1.writelines([x+"\n" for x in new_wavs])
        f2.writelines([x+"\n" for x in new_lid])