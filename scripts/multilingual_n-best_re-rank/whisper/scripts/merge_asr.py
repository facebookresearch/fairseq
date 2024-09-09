import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--splits', type=int)
    args = parser.parse_args()

    ltr = []
    wrd = []
    score = []
    length = []
    for i in range(args.splits):
        ltr += open(f"{args.src}/split_{str(i)}/hypo.ltr.reord", "r").readlines()
        wrd += open(f"{args.src}/split_{str(i)}/hypo.wrd.reord", "r").readlines()
        score += open(f"{args.src}/split_{str(i)}/score", "r").readlines()
        # length += open(f"{args.src}/split_{str(i)}/length", "r").readlines()

    with open(args.src + "/hypo.ltr.reord", "w") as f1, \
        open(args.src + "/hypo.wrd.reord", "w") as f2, \
        open(args.src + "/score", "w") as f3, \
        open(args.src + "/length", "w") as f4:
        f1.writelines(ltr)
        f2.writelines(wrd)
        f3.writelines(score)
        # f4.writelines(length)